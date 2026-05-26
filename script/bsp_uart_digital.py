import pygame
from PIL import Image, ImageOps
import numpy as np
import crcmod
import os
import time
import serial

# ===================== Settings =====================
WINDOW_SIZE = 512
TARGET_SIZE = 28  # <--- MAKE SURE THIS LINE EXISTS AND IS NOT COMMENTED OUT
PANEL_HEIGHT = 100
TOTAL_HEIGHT = WINDOW_SIZE + PANEL_HEIGHT

# Color definitions
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GRAY_PANEL = (240, 240, 240)  # Background color for control panel
GRAY_LINE = (200, 200, 200)  # Divider line color
GRAY_BTN = (100, 100, 100)
BTN_TEXT_COLOR = (255, 255, 255)

SERIAL_PORT = "/dev/ttyUSB0"
SERIAL_BAUD = 115200
FRAME_HEAD = 0xAA
FRAME_TAIL = 0x55

# Button Position and Size
# Panel area Y: 512 ~ 612. Center Y is 562. Button height is 40, so top Y is 542.
CLEAR_BTN = pygame.Rect(50, 542, 180, 40)
SEND_BTN = pygame.Rect(282, 542, 180, 40)
BRUSH_RADIUS = 12

# ===================== Initialize =====================
pygame.init()
screen = pygame.display.set_mode((WINDOW_SIZE, TOTAL_HEIGHT))
pygame.display.set_caption("MNIST Digit Drawer - Split Layout")
font = pygame.font.Font(None, 32)

# Serial Port Initialize
ser = None
try:
    port = SERIAL_PORT
    if port:
        ser = serial.Serial(port, SERIAL_BAUD, timeout=0.1)
        print(f"Serial initialized successfully on {port}")
    else:
        print("No serial device detected")
except Exception as e:
    print("Failed to open serial port:", e)

# Fill main canvas with pure white
screen.fill(WHITE)
drawing = False
last_pos = None
clock = pygame.time.Clock()
running = True

if not os.path.exists("picture"):
    os.makedirs("picture")

# ===================== Create Soft Brush =====================
# Create a soft brush surface with an alpha channel gradient (Gaussian feathering effect)
BRUSH_SIZE = BRUSH_RADIUS * 4
soft_brush = pygame.Surface((BRUSH_SIZE, BRUSH_SIZE), pygame.SRCALPHA)

for x in range(BRUSH_SIZE):
    for y in range(BRUSH_SIZE):
        dist = np.sqrt((x - BRUSH_SIZE / 2) ** 2 + (y - BRUSH_SIZE / 2) ** 2)
        max_dist = BRUSH_SIZE / 2
        if dist < max_dist:
            # Quadratic curve to simulate Gaussian feathering: Center (dist=0) alpha is 255, edge is 0
            alpha = int(255 * (1 - (dist / max_dist) ** 1.8))
            soft_brush.set_at((x, y), (0, 0, 0, max(0, min(255, alpha))))


# ===================== CRC16 Calculation =====================
def calc_crc16(data_bytes):
    crc16 = crcmod.Crc(0x18005, initCrc=0xFFFF, rev=True, xorOut=0x0000)
    crc16.update(data_bytes)
    return crc16.crcValue


# ===================== Pack Protocol Frame =====================
def pack_protocol_frame(gray_np):
    data_arr = gray_np.flatten().astype(np.uint8)
    data_bytes = data_arr.tobytes()
    data_len = len(data_bytes)

    frame = bytearray()
    frame.append(FRAME_HEAD)
    frame.append(data_len >> 8)
    frame.append(data_len & 0xFF)
    frame.extend(data_bytes)

    crc_val = calc_crc16(data_bytes)
    frame.append(crc_val & 0xFF)
    frame.append((crc_val >> 8) & 0xFF)
    frame.append(FRAME_TAIL)
    return frame


# ===================== Draw Panel & Buttons =====================
def draw_bottom_panel():
    # 1. Draw the control panel background (covers any brush lines drawn out of bounds)
    pygame.draw.rect(screen, GRAY_PANEL, (0, WINDOW_SIZE, WINDOW_SIZE, PANEL_HEIGHT))

    # 2. Draw the visual divider between canvas and panel
    pygame.draw.line(screen, GRAY_LINE, (0, WINDOW_SIZE), (WINDOW_SIZE, WINDOW_SIZE), 2)

    # 3. Clear Button
    pygame.draw.rect(screen, GRAY_BTN, CLEAR_BTN)
    clear_txt = font.render("Clear Canvas", True, BTN_TEXT_COLOR)
    text_rect = clear_txt.get_rect(center=CLEAR_BTN.center)
    screen.blit(clear_txt, text_rect)

    # 4. Send Button
    pygame.draw.rect(screen, GRAY_BTN, SEND_BTN)
    send_txt = font.render("Send to MCU", True, BTN_TEXT_COLOR)
    text_rect = send_txt.get_rect(center=SEND_BTN.center)
    screen.blit(send_txt, text_rect)


# ===================== Main Loop =====================
while running:
    # Redraw bottom panel on every frame to ensure layout isolation
    draw_bottom_panel()

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

        if event.type == pygame.MOUSEBUTTONDOWN:
            pos = event.pos
            # Check if click lands on the control panel buttons
            if CLEAR_BTN.collidepoint(pos):
                # Clear only the upper 512x512 canvas area to pure white
                pygame.draw.rect(screen, WHITE, (0, 0, WINDOW_SIZE, WINDOW_SIZE))
                if ser and ser.is_open:
                    ser.reset_input_buffer()
                    ser.reset_output_buffer()
                print("Canvas and serial buffers cleared")
            elif SEND_BTN.collidepoint(pos):
                # 1. Capture the upper 512x512 canvas area specifically to avoid capturing buttons
                pygame_surface = pygame.Surface((WINDOW_SIZE, WINDOW_SIZE))
                pygame_surface.blit(screen, (0, 0), (0, 0, WINDOW_SIZE, WINDOW_SIZE))
                pil_img = Image.frombytes(
                    "RGB",
                    (WINDOW_SIZE, WINDOW_SIZE),
                    pygame.image.tostring(pygame_surface, "RGB")
                ).convert("L")

                # 2. Downsample to 28x28 and invert colors (using BILINEAR to keep smooth gradient edges)
                img_28 = pil_img.resize((TARGET_SIZE, TARGET_SIZE), Image.Resampling.BILINEAR)
                img_28 = ImageOps.invert(img_28)
                gray_data = np.array(img_28)

                # 3. Save processed image
                timestamp = int(time.time() * 1000)
                save_path = os.path.join("picture", f"{timestamp}.png")
                img_28.save(save_path)
                print(f"28x28 image saved to: {save_path}")

                # 4. Package and transmit data
                send_frame = pack_protocol_frame(gray_data)

                # ASCII grayscale matrix preview in console (Dark -> 1, Light -> ., Empty -> 0)
                print("\n--- Image Matrix Preview ---")
                for line in gray_data:
                    for c in line:
                        if c > 180:
                            print("1", end="")
                        elif c > 40:
                            print(".", end="")
                        else:
                            print("0", end="")
                    print("")

                print(f"Frame length: {len(send_frame)}")

                if ser and ser.is_open:
                    ser.write(send_frame)
                    print("Frame sent to MCU successfully\n")
                else:
                    print("Serial not connected, cannot send\n")
            else:
                # Boundary check: Only start drawing if click is within the 512px canvas space
                if pos[1] < WINDOW_SIZE:
                    drawing = True
                    last_pos = pos
                    screen.blit(soft_brush, (pos[0] - BRUSH_SIZE // 2, pos[1] - BRUSH_SIZE // 2))

        if event.type == pygame.MOUSEBUTTONUP:
            drawing = False
            last_pos = None

        if event.type == pygame.MOUSEMOTION and drawing:
            curr_pos = event.pos
            # Anti-overflow truncation: Terminate drawing if mouse slides into the bottom panel area
            if curr_pos[1] >= WINDOW_SIZE:
                drawing = False
                last_pos = None
            elif last_pos:
                # High-density linear interpolation to prevent broken lines during rapid mouse movements
                points_dist = np.hypot(curr_pos[0] - last_pos[0], curr_pos[1] - last_pos[1])
                steps = int(points_dist / 2)  # Place a brush stamp every 2 pixels
                for i in range(max(1, steps)):
                    t = i / max(1, steps)
                    x = int(last_pos[0] + (curr_pos[0] - last_pos[0]) * t)
                    y = int(last_pos[1] + (curr_pos[1] - last_pos[1]) * t)
                    # Double check brush edges don't bleed into the bottom control panel
                    if y < WINDOW_SIZE:
                        screen.blit(soft_brush, (x - BRUSH_SIZE // 2, y - BRUSH_SIZE // 2))
                last_pos = curr_pos

    pygame.display.flip()
    clock.tick(60)

if ser and ser.is_open:
    ser.close()
pygame.quit()
print("Program exited, serial closed")
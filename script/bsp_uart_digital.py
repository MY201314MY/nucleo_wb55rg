import pygame
from PIL import Image, ImageOps
import numpy as np
import crcmod
import os
import time
import serial

# ===================== Settings =====================
WINDOW_SIZE = 512
TARGET_SIZE = 28  # MNIST 标准输入大小
PANEL_HEIGHT = 100
TOTAL_HEIGHT = WINDOW_SIZE + PANEL_HEIGHT

# Color definitions
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GRAY_PANEL = (240, 240, 240)  # 控制面板背景色
GRAY_LINE = (200, 200, 200)  # 画布与面板的分隔线
GRAY_BTN = (100, 100, 100)
BTN_TEXT_COLOR = (255, 255, 255)

SERIAL_PORT = "/dev/ttyUSB0"
SERIAL_BAUD = 115200
FRAME_HEAD = 0xAA
FRAME_TAIL = 0x55

# Button Position and Size
CLEAR_BTN = pygame.Rect(50, 542, 180, 40)
SEND_BTN = pygame.Rect(282, 542, 180, 40)
BRUSH_RADIUS = 12

# ===================== Initialize =====================
pygame.init()
screen = pygame.display.set_mode((WINDOW_SIZE, TOTAL_HEIGHT))
pygame.display.set_caption("MNIST Digit Drawer - Auto Centered Version")
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

# 初始填充画布为纯白
screen.fill(WHITE)
drawing = False
last_pos = None
clock = pygame.time.Clock()
running = True

if not os.path.exists("picture"):
    os.makedirs("picture")

# ===================== Create Soft Brush =====================
# 创建带有 Alpha 通道渐变的羽化软画笔，边缘过渡更平滑
BRUSH_SIZE = BRUSH_RADIUS * 4
soft_brush = pygame.Surface((BRUSH_SIZE, BRUSH_SIZE), pygame.SRCALPHA)

for x in range(BRUSH_SIZE):
    for y in range(BRUSH_SIZE):
        dist = np.sqrt((x - BRUSH_SIZE / 2) ** 2 + (y - BRUSH_SIZE / 2) ** 2)
        max_dist = BRUSH_SIZE / 2
        if dist < max_dist:
            # 二次曲线模拟高斯羽化：中心 alpha 为 255，边缘渐变为 0
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
    # 1. 绘制控制面板背景（遮挡可能溢出的画笔线条）
    pygame.draw.rect(screen, GRAY_PANEL, (0, WINDOW_SIZE, WINDOW_SIZE, PANEL_HEIGHT))

    # 2. 绘制画布和面板的分割线
    pygame.draw.line(screen, GRAY_LINE, (0, WINDOW_SIZE), (WINDOW_SIZE, WINDOW_SIZE), 2)

    # 3. 清除按钮
    pygame.draw.rect(screen, GRAY_BTN, CLEAR_BTN)
    clear_txt = font.render("Clear Canvas", True, BTN_TEXT_COLOR)
    text_rect = clear_txt.get_rect(center=CLEAR_BTN.center)
    screen.blit(clear_txt, text_rect)

    # 4. 发送按钮
    pygame.draw.rect(screen, GRAY_BTN, SEND_BTN)
    send_txt = font.render("Send to MCU", True, BTN_TEXT_COLOR)
    text_rect = send_txt.get_rect(center=SEND_BTN.center)
    screen.blit(send_txt, text_rect)


# ===================== Main Loop =====================
while running:
    # 每一帧都重绘底部面板，保持布局隔离
    draw_bottom_panel()

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

        if event.type == pygame.MOUSEBUTTONDOWN:
            pos = event.pos
            # 按钮事件响应
            if CLEAR_BTN.collidepoint(pos):
                # 仅擦除上方 512x512 绘图区域到纯白
                pygame.draw.rect(screen, WHITE, (0, 0, WINDOW_SIZE, WINDOW_SIZE))
                if ser and ser.is_open:
                    ser.reset_input_buffer()
                    ser.reset_output_buffer()
                print("Canvas and serial buffers cleared")

            elif SEND_BTN.collidepoint(pos):
                # 1. 截取上方 512x512 绘图区域并转为 L 灰度图
                pygame_surface = pygame.Surface((WINDOW_SIZE, WINDOW_SIZE))
                pygame_surface.blit(screen, (0, 0), (0, 0, WINDOW_SIZE, WINDOW_SIZE))
                pil_img = Image.frombytes(
                    "RGB",
                    (WINDOW_SIZE, WINDOW_SIZE),
                    pygame.image.tostring(pygame_surface, "RGB")
                ).convert("L")

                # 转换为黑底白字（符合标准神经网络的训练图像输入习惯）
                pil_img = ImageOps.invert(pil_img)

                # =====================================================================
                # 💡 核心改进：手写数字智能边界盒检测与绝对上下左右居中
                # =====================================================================
                gray_np_full = np.array(pil_img)

                # 寻找所有像素值大于安全阈值（例如 20）的坐标点，有效过滤边缘弱噪声
                y_indices, x_indices = np.where(gray_np_full > 20)

                if len(x_indices) > 0 and len(y_indices) > 0:
                    # 计算数字在 512x512 画布上的紧凑边界盒 (Bounding Box)
                    x_min, x_max = np.min(x_indices), np.max(x_indices)
                    y_min, y_max = np.min(y_indices), np.max(y_indices)

                    # 从原始大画布中裁剪出纯粹的数字字符区域
                    cropped_digit = pil_img.crop((x_min, y_min, x_max + 1, y_max + 1))
                    digit_w, digit_h = cropped_digit.size

                    # 为了使字符契合 MNIST 标准留白，限制长边等比例缩放到 20 像素
                    # （这样可以在 28x28 画布四周各留出约 4 像素的安全无信息边界）
                    max_side = max(digit_w, digit_h)
                    scale_ratio = 20.0 / max_side
                    new_w = int(digit_w * scale_ratio)
                    new_h = int(digit_h * scale_ratio)

                    # 缩放数字（使用 BILINEAR 滤波器保持渐变笔画平滑）
                    digit_resized = cropped_digit.resize((new_w, new_h), Image.Resampling.BILINEAR)

                    # 创建标准的 28x28 纯黑底色新画布
                    img_28 = Image.new("L", (TARGET_SIZE, TARGET_SIZE), 0)

                    # 计算几何绝对居中粘贴所需的 X, Y 轴像素偏移量
                    paste_x = (TARGET_SIZE - new_w) // 2
                    paste_y = (TARGET_SIZE - new_h) // 2

                    # 粘贴进新画布中心
                    img_28.paste(digit_resized, (paste_x, paste_y))
                    print(f"✓ 检测到有效笔画，已成功执行绝对几何居中 (Offset X:{paste_x}, Y:{paste_y})")
                else:
                    # 全白画布（未书写任何内容）默认返回纯黑底图数据
                    print("⚠ 未检测到有效笔画，下发全空数据")
                    img_28 = Image.new("L", (TARGET_SIZE, TARGET_SIZE), 0)

                # 将最终完美居中的图像转换为一维/二维 Numpy 矩阵
                gray_data = np.array(img_28)
                # =====================================================================

                # 3. 保存处理完成的 28x28 完美居中测试图
                timestamp = int(time.time() * 1000)
                save_path = os.path.join("picture", f"{timestamp}.png")
                img_28.save(save_path)
                print(f"28x28 centered image saved to: {save_path}")

                # 4. 封装协议数据帧
                send_frame = pack_protocol_frame(gray_data)

                # 终端控制台 ASCII 字符矩阵预览（可用肉眼直观核对是否完美居中）
                print("\n--- Centered Image Matrix Preview ---")
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
                # 边界保护：点击必须在 512px 绘图区内才触发画笔
                if pos[1] < WINDOW_SIZE:
                    drawing = True
                    last_pos = pos
                    screen.blit(soft_brush, (pos[0] - BRUSH_SIZE // 2, pos[1] - BRUSH_SIZE // 2))

        if event.type == pygame.MOUSEBUTTONUP:
            drawing = False
            last_pos = None

        if event.type == pygame.MOUSEMOTION and drawing:
            curr_pos = event.pos
            # 防溢出截断：当鼠标滑出绘图区进入底部面板，强制终止书写
            if curr_pos[1] >= WINDOW_SIZE:
                drawing = False
                last_pos = None
            elif last_pos:
                # 高密度线性插值，杜绝因快速挥动鼠标而产生“断墨”点阵
                points_dist = np.hypot(curr_pos[0] - last_pos[0], curr_pos[1] - last_pos[1])
                steps = int(points_dist / 2)  # 每 2 像素高密度盖印一个画笔戳
                for i in range(max(1, steps)):
                    t = i / max(1, steps)
                    x = int(last_pos[0] + (curr_pos[0] - last_pos[0]) * t)
                    y = int(last_pos[1] + (curr_pos[1] - last_pos[1]) * t)
                    # 二次保障：画笔戳边缘绝不污染底部控制面板
                    if y < WINDOW_SIZE:
                        screen.blit(soft_brush, (x - BRUSH_SIZE // 2, y - BRUSH_SIZE // 2))
                last_pos = curr_pos

    pygame.display.flip()
    clock.tick(60)

if ser and ser.is_open:
    ser.close()
pygame.quit()
print("Program exited, serial closed")
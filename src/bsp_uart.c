#include <zephyr/kernel.h>
#include <zephyr/device.h>
#include <zephyr/drivers/uart.h>
#include <zephyr/shell/shell.h>

#include "bsp_uart.h"

#include <zephyr/logging/log.h>
LOG_MODULE_REGISTER( uart, LOG_LEVEL_INF );

static uint8_t uart_rxbuffer[1024] = { 0 };
static struct ring_buf rx_rb;

static uint8_t uart_txbuffer[8] = { 0 };
static struct ring_buf tx_rb;

static struct k_mutex rx_mutex;
static struct k_mutex tx_mutex;

static struct k_timer rx_timer = {0};

#define RX_TIMEOUT_MS 20

struct k_sem bsp_rx_sem = {0};
uint8_t bsp_rx_buffer[1024] = { 0 };
uint8_t digital_buffer[784] = { 0 };

static void cb_handler_rx( const struct device *dev )
{
    uint8_t rxbuffer[32] = { 0 };
    int size             = uart_fifo_read( dev, rxbuffer, sizeof( rxbuffer ) );

    if( size <= 0 )
    {
        LOG_ERR( "failed to read uart: %d", size );
        return;
    }

    ring_buf_put( &rx_rb, rxbuffer, size );
    k_timer_start( &rx_timer, K_MSEC( RX_TIMEOUT_MS ), K_NO_WAIT );

    LOG_HEXDUMP_DBG( rxbuffer, size, "uart received" );
}

static void cb_handler_tx( const struct device *dev )
{
    uint8_t txbuffer[32] = { 0 };
    uint32_t cnt         = ring_buf_get( &tx_rb, txbuffer, sizeof( txbuffer ) );

    if( cnt > 0 )
    {
        uart_fifo_fill( dev, txbuffer, cnt );
    }
    else
    if( uart_irq_tx_complete( dev ) )
    {
        uart_irq_tx_disable( dev );
    }
}

static void uart_cb_handler( const struct device *dev, void *user_data )
{
    if( uart_irq_update( dev ) && uart_irq_is_pending( dev ) )
    {
        if( uart_irq_rx_ready( dev ) )
        {
        cb_handler_rx( dev );
        }

        if( uart_irq_tx_ready( dev ) )
        {
            cb_handler_tx( dev );
        }
    }
}

void rx_timer_handler( struct k_timer *timer_id )
{
    int ret = ring_buf_get( &rx_rb, bsp_rx_buffer, sizeof( bsp_rx_buffer ) );
    LOG_INF( "rx_timer_handler: got %d bytes from ring buffer", ret );

    /*
        28*28 = 784
        0 : header
        1-2 : length (LSB, MSB)
        3-786 : data
        787-788 : checksum (LSB, MSB)
        789 : tail 
    */

    if(ret == 790)
    {
        memcpy(digital_buffer, bsp_rx_buffer + 3, 784);
        k_sem_give( &bsp_rx_sem );
    }
}

int bsp_uart_init( void )
{
    const struct device *uart = DEVICE_DT_GET( DT_NODELABEL( lpuart1 ) );

    k_mutex_init( &rx_mutex );
    k_mutex_init( &tx_mutex );

    k_sem_init( &bsp_rx_sem, 0, 1 );

    ring_buf_init( &rx_rb, sizeof( uart_rxbuffer ), uart_rxbuffer );
    ring_buf_init( &tx_rb, sizeof( uart_txbuffer ), uart_txbuffer );
    uart_irq_callback_user_data_set( uart, uart_cb_handler, NULL );
    uart_irq_rx_enable( uart );
    uart_irq_tx_disable( uart );

    k_timer_init( &rx_timer, rx_timer_handler, NULL );

  return 0;
}


SYS_INIT( bsp_uart_init, APPLICATION, 99 );


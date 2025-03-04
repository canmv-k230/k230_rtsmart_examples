/**
 * Copyright (c) 2023, Canaan Bright Sight Co., Ltd
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 * 1. Redistributions of source code must retain the above copyright
 * notice, this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright
 * notice, this list of conditions and the following disclaimer in the
 * documentation and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND
 * CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES,
 * INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
 * MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 * SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 * BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
 * WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
 * NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

 #include<stdio.h>
 #include <stdio.h>
 #include <stdlib.h>
 #include <string.h>
 #include <pthread.h>
 #include <fcntl.h>
 #include <unistd.h>

 #include <rtthread.h>
 #include <rtdevice.h>


 #define RT_I2C_DEV_CTRL_10BIT (0x800 + 0x01)
 #define RT_I2C_DEV_CTRL_TIMEOUT (0x800 + 0x03)
 #define RT_I2C_DEV_CTRL_RW (0x800 + 0x04)
 #define RT_I2C_DEV_CTRL_CLK (0x800 + 0x05)

 enum dm_i2c_msg_flags
 {
	 I2C_M_TEN           = 0x0010, /* ten-bit chip address */
	 I2C_M_RD            = 0x0001, /* read data, from slave to master */
	 I2C_M_STOP          = 0x8000, /* send stop after this message */
	 I2C_M_NOSTART       = 0x4000, /* no start before this message */
	 I2C_M_REV_DIR_ADDR  = 0x2000, /* invert polarity of R/W bit */
	 I2C_M_IGNORE_NAK    = 0x1000, /* continue after NAK */
	 I2C_M_NO_RD_ACK     = 0x0800, /* skip the Ack bit on reads */
	 I2C_M_RECV_LEN      = 0x0400, /* length is first received byte */
	 I2C_M_RESTART       = 0x0002, /* restart before this message */
	 I2C_M_START         = 0x0004, /* start before this message */
 };



 typedef struct {
	 uint16_t addr;
	 uint16_t flags;
	 uint16_t len;
	 uint8_t *buf;
 } i2c_msg_t;

 typedef struct {
	 i2c_msg_t *msgs;
	 size_t number;
 } i2c_priv_data_t;


 int  main(int argc, char *argv[])
 {
	int fd;
	char *name = "/dev/i2c4";
	rt_uint32_t spped=100000;

	if (argc >= 2)
		name = argv[1];

	if (argc >= 3)
		spped = atoi(argv[2]);

	printf("use device name=%s speed=%d\n", name, spped);
	fd = open(name, O_RDWR);
	if (fd < 0 ){
		printf(" %s device open failed\n", name);
		return -1;
	}

	if (ioctl(fd, RT_I2C_DEV_CTRL_CLK, &spped) != RT_EOK){
		printf("set %s speed %d failed!\n", name, spped);
	}

	for (int i = 0x08; i < 0x78; i++){
		unsigned char data;
		i2c_msg_t msgs={i, I2C_M_RD, 1, &data};
		i2c_priv_data_t privdata={&msgs, 1};

		if (ioctl(fd, RT_I2C_DEV_CTRL_RW, &privdata) == 0){
			printf("find device at add 0x%02hx and data=%x\n", i, data);
		}
	}

	close(fd);
 }

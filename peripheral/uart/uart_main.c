/* Copyright (c) 2025, Canaan Bright Sight Co., Ltd
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

 #include <stdio.h>
 #include <stdlib.h>
 #include <unistd.h>
 #include <string.h>
 #include <sys/mman.h>
 #include <pthread.h>
 #include <fcntl.h>
 #include <rtthread.h>
 #include <rtdevice.h>
 #include <poll.h>


 #define	IOC_SET_BAUDRATE            _IOW('U', 0x40, int)

 struct uart_configure
 {
	 rt_uint32_t baud_rate;

	 rt_uint32_t data_bits               :4;
	 rt_uint32_t stop_bits               :2;
	 rt_uint32_t parity                  :2;
	 rt_uint32_t fifo_lenth              :2;
	 rt_uint32_t auto_flow               :1;
	 rt_uint32_t reserved                :21;
 };

 typedef enum _uart_parity
 {
	 UART_PARITY_NONE,
	 UART_PARITY_ODD,
	 UART_PARITY_EVEN
 } uart_parity_t;

 typedef enum _uart_receive_trigger
 {
	 UART_RECEIVE_FIFO_1,
	 UART_RECEIVE_FIFO_8,
	 UART_RECEIVE_FIFO_16,
	 UART_RECEIVE_FIFO_30,
 } uart_receive_trigger_t;
 #define SEND_START_STR "uart test start !\n"
 int main(int argc,char *argv[])
 {
	int fd;
	int ret = 0;
	char *dev_name = "/dev/uart3";

	if(argc > 1)
		dev_name = argv[1];

	fd = open(dev_name, O_RDWR);
	if (fd < 0){
		printf("open dev %s  failed!\n", dev_name);
		return 1;
	}
	printf("open dev %s ok \n", dev_name);

	struct uart_configure config = {
		.baud_rate = 9600,
		.data_bits = 8,
		.stop_bits = 1,
		.parity = UART_PARITY_NONE,
		.fifo_lenth = UART_RECEIVE_FIFO_16,
		.auto_flow = 0,
	};
	if (ioctl(fd, IOC_SET_BAUDRATE, &config)) {
		printf("ioctl failed!\n");
		close(fd);
		return 1;
	}
	write(fd, SEND_START_STR, strlen(SEND_START_STR));
	while(1) {
		fd_set rfds;
		int retval;

		FD_ZERO(&rfds);
		FD_SET(fd, &rfds); // 监视标准输入流

		retval = select(fd+1, &rfds, NULL, NULL, NULL);
		if (retval < 0) {
			printf("select failed");
			break;
		}
		if(FD_ISSET(fd, &rfds)){
			char buf[256];
			memset(buf, 0, sizeof(buf));
			ret = read(fd, buf, sizeof(buf));
			printf("ret =%x read :%s\n",ret, buf);
			if(strchr(buf,'q'))
				break;
		}
	}


	close(fd);
	return 0;
 }

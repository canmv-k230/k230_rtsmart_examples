/**
 * Copyright (c) 2025, Canaan Bright Sight Co., Ltd
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
 #include <dfs_poll.h>


 #define I2C_SLAVE_IOCTL_SET_BUFFER_SIZE               0
 #define I2C_SLAVE_IOCTL_SET_ADDR                      1




 int  main(int argc, char *argv[])
 {
	int fd;
    char add = 0x22;
    int size = 20;
    int ret = 0;
    if(argc < 2) {
        printf("examlpe: i2c_slave.elf /dev/i2c1_slave [slave_add] [dev_size] [data]]\n");
        printf("examlpe: i2c_slave.elf  /dev/i2c1_slave 33  32 90\n");
        return 1;
    }
    fd = open(argv[1], O_RDWR);
    if (fd < 0) {
        printf("Can't open %s\n", argv[1]);
        return 1;
    }
    if(argc >= 3) {
        add = atoi(argv[2]);
        ret = ioctl(fd, I2C_SLAVE_IOCTL_SET_ADDR, &add);
        if (ret) {
            printf("set add  %x failed!\n", add);
            return 1;
        }
    }
    if(argc >= 4) {
        size = atoi(argv[3]);
        ret = ioctl(fd, I2C_SLAVE_IOCTL_SET_BUFFER_SIZE, &size);
        if (ret) {
            printf("set size  %d failed!\n", size);
            return 1 ;
        }
    }

    if(argc >= 5) {
        unsigned char  data = atoi(argv[4]);
        lseek(fd, 0,SEEK_SET);
        for(int i = 0; i < size; i++,data++)
            write(fd, &data, 1);
    }


    {
        char buffer[16];
        int size = 0;
        lseek(fd, 0,SEEK_SET);
        //do {

            size = read(fd, buffer, sizeof(buffer));
            for (unsigned i = 0; i < size; i++) {
                printf("%02x ", buffer[i]);
            }
            printf("\n");
        //} while (size != 0);
    }

    {
        struct pollfd pfd;
        pfd.fd = fd;
        pfd.events = POLLIN;
        ret = poll(&pfd, 1, -1);
    }


    //printf("ret=%x\n",ret);
    usleep(1000);
    //printf("func =%s l=%d \n", __func__, __LINE__);
    {
        char buffer[16];
        int size = 0;
        lseek(fd, 0,SEEK_SET);
        //do {

            size = read(fd, buffer, sizeof(buffer));
            printf("func =%s l=%d size=%X  \n", __func__, __LINE__, size);
            for (unsigned i = 0; i < size; i++) {
                printf("%02x ", buffer[i]);
            }
            printf("\n");
        //} while (size != 0);
    }



    printf("i2c_slave_addr: 0x%02x, mem_size: %d\r\n",add,size);
    usleep(1000);
    close(fd);
    return 0;
}

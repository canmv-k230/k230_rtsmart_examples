# RT-Smart socket compatibility test

This test uses a native Linux peer and a K230 test process. It covers the
RT-Smart IPv4 socket API, including TCP and UDP, blocking and nonblocking I/O,
`select`, `poll`, `ioctl(FIONREAD)`, socket options, name queries, `shutdown`,
`sendmsg`, `recvmsg`, `sendmmsg`, `recvmmsg`, and `accept4`.

Build both executables:

```sh
make -C src/rtsmart/examples/test/socket_compat
```

Run the peer on the PC (allow TCP and UDP port 5202 through the firewall):

```sh
output/k230_canmv_01studio_defconfig/rtsmart/examples/test/socket_compat/socket_compat_peer
```

Copy `src/rtsmart/examples/elf/test/socket_compat.elf` to the board and run:

```text
msh />/sdcard/socket_compat.elf <PC-IP> 5202
```

The final line reports pass, fail, and skip counts. A skip is used only for a
compile-time lwIP feature such as `LWIP_NETBUF_RECVINFO`.

The suite treats the following RT-Smart policies as expected results:

- IPv6 is disabled by the current BSP configuration.
- Unix-domain `socketpair` is outside the lwIP/SAL IPv4 backend.
- `SOCK_CLOEXEC` is not supported because RT-Smart does not yet track
  close-on-exec state per descriptor.
- Sending ancillary control messages and Linux UDP GSO/GRO are not lwIP
  features. Receiving IPv4 `IP_PKTINFO` is supported when
  `LWIP_NETBUF_RECVINFO` is enabled.

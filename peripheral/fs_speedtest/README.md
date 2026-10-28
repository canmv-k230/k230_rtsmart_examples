# Filesystem speed and integrity test

Build with `make -C src/rtsmart/examples/peripheral/fs_speedtest` from the SDK
root after building the configured board's libraries. The executable is
`src/rtsmart/examples/elf/peripheral/fs_speedtest.elf`.

On the board, select a directory on the eMMC or SD filesystem to test:

```sh
fs_speedtest.elf /sdcard
```

With no arguments, the test uses `/sdcard` and `/data`. Multiple directory
arguments are accepted. The benchmark creates, overwrites and removes
`helloworld_fsbench.bin` in each directory and needs at least 8 MiB of free space.

For each block size (16 KiB, 64 KiB, 256 KiB and 1 MiB), it tests sequential and
shuffled writes, both with and without preallocation. Every write case is
flushed with `fsync`, closed, reopened and read back before the next case starts.
Data varies with its file offset and with each case, allowing misplaced or stale
blocks to be detected. The existing fill and soak modes also use the stronger
offset pattern.

Each case prints `write_sha256`, `read_sha256` and `SHA-256 PASS` or `SHA-256 FAIL`.
The two digests must match. Shuffled cases hash blocks in the same shuffled order
for writing and reading, so their digests are not the sequential file digest
produced by `sha256sum`. A hash mismatch, incorrect file size or I/O error stops
the benchmark with a nonzero exit status.

Reported I/O time includes seeks, reads/writes and the write flush. Pattern
generation and hashing are excluded, so total runtime is longer than the
reported I/O time. Reopening the file does not guarantee that every filesystem
bypasses its caches. Run on the intended eMMC mount after confirming HS200 in
the board's boot log; this program does not select or verify the MMC timing mode.

The existing `--fill-verify <mount_point> [percent]` and
`--soak <mount_point> <seconds> [file_kib]` modes remain available; use `--help`
for the command syntax.

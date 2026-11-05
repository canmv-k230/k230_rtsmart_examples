/* Copyright (c) 2026, Canaan Bright Sight Co., Ltd
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
#ifndef SOFT_GZIP_H
#define SOFT_GZIP_H

#include <stdint.h>

/* A small gzip encoder for the self test: LZ77 plus one dynamic-Huffman
 * table per block, which is what the K230 engine decodes. The output is
 * marked hardware-compatible (method byte 0x09, as K230PrivGzip writes it)
 * and has the fixed header the engine expects: no name, mtime 0, OS 3. */
typedef struct {
    uint32_t window;       /* longest match distance, a power of two <= 32768 */
    uint32_t max_match;    /* longest match length, 3..258 */
    uint32_t block_tokens; /* literals and matches per deflate block */
} soft_gzip_opt_t;

/* Returns a malloc'ed stream and its length in *out_len, or NULL. */
uint8_t *soft_gzip_compress(const uint8_t *in, uint32_t len, const soft_gzip_opt_t *opt,
                            uint32_t *out_len);

#endif /* SOFT_GZIP_H */

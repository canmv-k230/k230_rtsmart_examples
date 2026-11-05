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
#include <stdlib.h>
#include <string.h>

#include "soft_gzip.h"

#define MIN_MATCH  3
#define HASH_BITS  15
#define HASH_SIZE  (1u << HASH_BITS)
#define MAX_CHAIN  64
#define NUM_LIT    286
#define NUM_DIST   30
#define NUM_CL     19
#define END_BLOCK  256

/* A token is a literal (bit 31 clear, byte in the low bits) or a match
 * (bit 31 set, length in bits 16..24, distance in bits 0..15). */
#define TOK_MATCH        0x80000000u
#define TOK(len, dist)   (TOK_MATCH | ((uint32_t)(len) << 16) | (uint32_t)(dist))

static const uint16_t len_base[29] = { 3,  4,  5,  6,  7,  8,  9,  10, 11,  13,  15,  17,  19,  23, 27,
                                       31, 35, 43, 51, 59, 67, 83, 99, 115, 131, 163, 195, 227, 258 };
static const uint8_t  len_extra[29] = { 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2,
                                        2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 0 };
static const uint16_t dist_base[30] = { 1,   2,   3,   4,   5,    7,    9,    13,   17,   25,
                                        33,  49,  65,  97,  129,  193,  257,  385,  513,  769,
                                        1025, 1537, 2049, 3073, 4097, 6145, 8193, 12289, 16385, 24577 };
static const uint8_t  dist_extra[30] = { 0, 0, 0, 0, 1, 1, 2, 2,  3,  3,  4,  4,  5,  5,  6,
                                         6, 7, 7, 8, 8, 9, 9, 10, 10, 11, 11, 12, 12, 13, 13 };
static const uint8_t  cl_order[NUM_CL] = { 16, 17, 18, 0, 8, 7, 9, 6, 10, 5, 11, 4, 12, 3, 13, 2, 14, 1, 15 };

typedef struct {
    uint8_t* buf;
    uint32_t cap;
    uint32_t len;
    uint64_t bits;
    int      nbits;
    int      oom;
} bitw_t;

static void bw_byte(bitw_t* w, uint8_t b)
{
    if (w->len == w->cap) {
        uint32_t cap = w->cap * 2 + 1024;
        uint8_t* buf = realloc(w->buf, cap);

        if (!buf) {
            w->oom = 1;
            return;
        }
        w->buf = buf;
        w->cap = cap;
    }
    w->buf[w->len++] = b;
}

/* Deflate packs bits least significant first. */
static void bw_put(bitw_t* w, uint32_t value, int nbits)
{
    w->bits |= (uint64_t)value << w->nbits;
    w->nbits += nbits;
    while (w->nbits >= 8) {
        bw_byte(w, (uint8_t)w->bits);
        w->bits >>= 8;
        w->nbits -= 8;
    }
}

static void bw_flush(bitw_t* w)
{
    if (w->nbits > 0)
        bw_byte(w, (uint8_t)w->bits);
    w->bits  = 0;
    w->nbits = 0;
}

static int len_code(uint32_t len)
{
    int i = 28;

    while (len_base[i] > len)
        i--;
    return i;
}

static int dist_code(uint32_t dist)
{
    int i = 29;

    while (dist_base[i] > dist)
        i--;
    return i;
}

/* Huffman code lengths for freq[], none longer than limit. Every symbol
 * with a nonzero count gets a code; at least two do, so the code is
 * complete as decoders expect. */
static void build_lengths(uint32_t* freq, int n, int limit, uint8_t* lens)
{
    uint32_t weight[2 * NUM_LIT];
    int      parent[2 * NUM_LIT];
    int      sym[NUM_LIT];
    int      count[16] = { 0 };
    int      live[2 * NUM_LIT];
    int      nsym = 0, nlive, nodes, i, j;
    uint32_t total;

    memset(lens, 0, n);
    for (i = 0; i < n && nsym < 2; i++) {
        if (freq[i])
            nsym++;
    }
    for (i = 0; nsym < 2 && i < n; i++) {
        if (!freq[i]) {
            freq[i] = 1;
            nsym++;
        }
    }

    nsym = 0;
    for (i = 0; i < n; i++) {
        if (freq[i]) {
            sym[nsym]    = i;
            weight[nsym] = freq[i];
            parent[nsym] = -1;
            live[nsym]   = nsym;
            nsym++;
        }
    }

    /* Plain Huffman: merge the two lightest live nodes until one is left. */
    nodes = nsym;
    nlive = nsym;
    while (nlive > 1) {
        int a = 0, b = 1, k;

        if (weight[live[b]] < weight[live[a]]) {
            a = 1;
            b = 0;
        }
        for (k = 2; k < nlive; k++) {
            if (weight[live[k]] < weight[live[a]]) {
                b = a;
                a = k;
            } else if (weight[live[k]] < weight[live[b]]) {
                b = k;
            }
        }
        weight[nodes]       = weight[live[a]] + weight[live[b]];
        parent[nodes]       = -1;
        parent[live[a]]     = nodes;
        parent[live[b]]     = nodes;
        live[a]             = nodes;
        live[b]             = live[--nlive];
        nodes++;
    }

    for (i = 0; i < nsym; i++) {
        int depth = 0;

        for (j = i; parent[j] >= 0; j = parent[j])
            depth++;
        count[depth > limit ? limit : depth]++;
    }

    /* Clamping long codes oversubscribes the code space: lengthen shorter
     * codes until the Kraft sum is exactly one again. Each pass lowers total
     * by one, and the excess is below the number of clamped symbols, so this
     * runs fewer than n passes and count[limit] never drops to zero. */
    total = 0;
    for (i = 1; i <= limit; i++)
        total += (uint32_t)count[i] << (limit - i);
    while (total > (1u << limit)) {
        count[limit]--;
        for (i = limit - 1; i > 0; i--) {
            if (count[i]) {
                count[i]--;
                count[i + 1] += 2;
                break;
            }
        }
        total--;
    }

    /* Most frequent symbols get the shortest codes. */
    for (i = 1; i < nsym; i++) {
        int s = sym[i];

        for (j = i; j > 0 && freq[sym[j - 1]] < freq[s]; j--)
            sym[j] = sym[j - 1];
        sym[j] = s;
    }
    j = 0;
    for (i = 1; i <= limit; i++) {
        int k;

        for (k = 0; k < count[i]; k++)
            lens[sym[j++]] = (uint8_t)i;
    }
}

/* Canonical codes, bit-reversed for least-significant-first output. */
static void build_codes(const uint8_t* lens, int n, uint16_t* codes)
{
    uint16_t next[16] = { 0 };
    int      count[16] = { 0 };
    uint16_t code = 0;
    int      i, b;

    for (i = 0; i < n; i++)
        count[lens[i]]++;
    count[0] = 0;
    for (b = 1; b < 16; b++) {
        code    = (uint16_t)((code + count[b - 1]) << 1);
        next[b] = code;
    }
    for (i = 0; i < n; i++) {
        uint16_t c = 0, v;

        if (!lens[i])
            continue;
        v = next[lens[i]]++;
        for (b = 0; b < lens[i]; b++)
            c |= (uint16_t)(((v >> b) & 1) << (lens[i] - 1 - b));
        codes[i] = c;
    }
}

/* Run-length code the literal and distance code lengths with symbols 16,
 * 17 and 18. Each output entry is symbol | extra bits << 8. Every entry
 * covers at least one length, so out[] needs at most n entries. */
static int rle_lengths(const uint8_t* lens, int n, uint16_t* out)
{
    int i = 0, m = 0;

    while (i < n) {
        uint8_t v = lens[i];
        int     run = 1;

        while (i + run < n && lens[i + run] == v)
            run++;
        i += run;
        if (v == 0) {
            while (run >= 11) {
                int k = run > 138 ? 138 : run;

                out[m++] = (uint16_t)(18 | ((k - 11) << 8));
                run -= k;
            }
            if (run >= 3) {
                out[m++] = (uint16_t)(17 | ((run - 3) << 8));
                run      = 0;
            }
        } else {
            out[m++] = v;
            run--;
            while (run >= 3) {
                int k = run > 6 ? 6 : run;

                out[m++] = (uint16_t)(16 | ((k - 3) << 8));
                run -= k;
            }
        }
        while (run-- > 0)
            out[m++] = v;
    }
    return m;
}

static void emit_block(bitw_t* w, const uint32_t* tok, uint32_t ntok, int final)
{
    uint32_t lit_freq[NUM_LIT] = { 0 }, dist_freq[NUM_DIST] = { 0 }, cl_freq[NUM_CL] = { 0 };
    uint8_t  lens[NUM_LIT + NUM_DIST], cl_lens[NUM_CL];
    uint16_t lit_codes[NUM_LIT], dist_codes[NUM_DIST], cl_codes[NUM_CL];
    uint16_t rle[NUM_LIT + NUM_DIST];
    int      hlit, hdist, hclen, nrle, i;
    uint32_t t;

    for (t = 0; t < ntok; t++) {
        if (tok[t] & TOK_MATCH) {
            lit_freq[257 + len_code((tok[t] >> 16) & 0x1ff)]++;
            dist_freq[dist_code(tok[t] & 0xffff)]++;
        } else {
            lit_freq[tok[t]]++;
        }
    }
    lit_freq[END_BLOCK]++;

    build_lengths(lit_freq, NUM_LIT, 15, lens);
    build_lengths(dist_freq, NUM_DIST, 15, lens + NUM_LIT);
    build_codes(lens, NUM_LIT, lit_codes);
    build_codes(lens + NUM_LIT, NUM_DIST, dist_codes);

    for (hlit = NUM_LIT; hlit > 257 && !lens[hlit - 1]; hlit--)
        ;
    for (hdist = NUM_DIST; hdist > 1 && !lens[NUM_LIT + hdist - 1]; hdist--)
        ;
    /* Literal and distance lengths form one sequence for the run-length
     * coder: move the distance lengths down behind the hlit used ones. */
    memmove(lens + hlit, lens + NUM_LIT, hdist);
    nrle = rle_lengths(lens, hlit + hdist, rle);
    memmove(lens + NUM_LIT, lens + hlit, hdist);

    for (i = 0; i < nrle; i++)
        cl_freq[rle[i] & 0xff]++;
    build_lengths(cl_freq, NUM_CL, 7, cl_lens);
    build_codes(cl_lens, NUM_CL, cl_codes);
    for (hclen = NUM_CL; hclen > 4 && !cl_lens[cl_order[hclen - 1]]; hclen--)
        ;

    bw_put(w, final ? 1 : 0, 1);
    bw_put(w, 2, 2); /* BTYPE 10: dynamic Huffman */
    bw_put(w, hlit - 257, 5);
    bw_put(w, hdist - 1, 5);
    bw_put(w, hclen - 4, 4);
    for (i = 0; i < hclen; i++)
        bw_put(w, cl_lens[cl_order[i]], 3);
    for (i = 0; i < nrle; i++) {
        int s = rle[i] & 0xff, extra = rle[i] >> 8;

        bw_put(w, cl_codes[s], cl_lens[s]);
        if (s == 16)
            bw_put(w, extra, 2);
        else if (s == 17)
            bw_put(w, extra, 3);
        else if (s == 18)
            bw_put(w, extra, 7);
    }

    for (t = 0; t < ntok; t++) {
        if (tok[t] & TOK_MATCH) {
            uint32_t len = (tok[t] >> 16) & 0x1ff, dist = tok[t] & 0xffff;
            int      lc = len_code(len), dc = dist_code(dist);

            bw_put(w, lit_codes[257 + lc], lens[257 + lc]);
            bw_put(w, len - len_base[lc], len_extra[lc]);
            bw_put(w, dist_codes[dc], lens[NUM_LIT + dc]);
            bw_put(w, dist - dist_base[dc], dist_extra[dc]);
        } else {
            bw_put(w, lit_codes[tok[t]], lens[tok[t]]);
        }
    }
    bw_put(w, lit_codes[END_BLOCK], lens[END_BLOCK]);
}

static uint32_t crc32_update(uint32_t crc, const uint8_t* p, uint32_t len)
{
    static uint32_t table[256];
    static int      ready;
    uint32_t        i;

    if (!ready) {
        for (i = 0; i < 256; i++) {
            uint32_t c = i;
            int      k;

            for (k = 0; k < 8; k++)
                c = (c & 1) ? 0xedb88320u ^ (c >> 1) : c >> 1;
            table[i] = c;
        }
        ready = 1;
    }
    crc = ~crc;
    for (i = 0; i < len; i++)
        crc = table[(crc ^ p[i]) & 0xff] ^ (crc >> 8);
    return ~crc;
}

static uint32_t hash3(const uint8_t* p) { return ((p[0] << 10) ^ (p[1] << 5) ^ p[2]) & (HASH_SIZE - 1); }

uint8_t* soft_gzip_compress(const uint8_t* in, uint32_t len, const soft_gzip_opt_t* opt, uint32_t* out_len)
{
    static const uint8_t header[10] = { 0x1f, 0x8b, 0x09, 0x00, 0, 0, 0, 0, 0x00, 0x03 };
    uint32_t window = opt->window, max_match = opt->max_match, block = opt->block_tokens;
    int32_t* head = NULL;
    int32_t* prev = NULL;
    uint32_t* tok = NULL;
    uint32_t  ntok = 0, pos = 0, crc, i;
    bitw_t    w = { 0 };

    if (!window || (window & (window - 1)) || window > 32768 || max_match < MIN_MATCH || max_match > 258 || !block)
        return NULL;

    head = malloc(HASH_SIZE * sizeof(*head));
    prev = malloc(window * sizeof(*prev));
    tok  = malloc(block * sizeof(*tok));
    if (!head || !prev || !tok)
        goto fail;
    memset(head, 0xff, HASH_SIZE * sizeof(*head));

    for (i = 0; i < sizeof(header); i++)
        bw_byte(&w, header[i]);

    while (pos < len) {
        uint32_t best_len = 0, best_dist = 0, step;

        if (pos + MIN_MATCH <= len) {
            uint32_t h     = hash3(in + pos);
            int32_t  cand  = head[h];
            int      chain = MAX_CHAIN;
            uint32_t limit = len - pos < max_match ? len - pos : max_match;

            while (cand >= 0 && chain-- > 0 && pos - (uint32_t)cand <= window) {
                uint32_t l = 0;

                while (l < limit && in[cand + l] == in[pos + l])
                    l++;
                if (l > best_len) {
                    best_len  = l;
                    best_dist = pos - (uint32_t)cand;
                    if (l == limit)
                        break;
                }
                cand = prev[cand & (window - 1)];
            }
        }

        if (best_len >= MIN_MATCH) {
            tok[ntok++] = TOK(best_len, best_dist);
            step        = best_len;
        } else {
            tok[ntok++] = in[pos];
            step        = 1;
        }
        for (i = 0; i < step; i++, pos++) {
            if (pos + MIN_MATCH <= len) {
                uint32_t h = hash3(in + pos);

                prev[pos & (window - 1)] = head[h];
                head[h]                  = (int32_t)pos;
            }
        }
        if (ntok == block) {
            emit_block(&w, tok, ntok, pos == len);
            ntok = 0;
        }
    }
    if (ntok || !len)
        emit_block(&w, tok, ntok, 1);
    bw_flush(&w);

    crc = crc32_update(0, in, len);
    for (i = 0; i < 4; i++)
        bw_byte(&w, (uint8_t)(crc >> (8 * i)));
    for (i = 0; i < 4; i++)
        bw_byte(&w, (uint8_t)(len >> (8 * i)));
    if (w.oom)
        goto fail;

    free(head);
    free(prev);
    free(tok);
    *out_len = w.len;
    return w.buf;

fail:
    free(head);
    free(prev);
    free(tok);
    free(w.buf);
    return NULL;
}

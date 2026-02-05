#pragma once

#include <cstdint>

#include "sage_vdb/anns/song/config.hpp"

namespace song_kernel {

template <typename T, int max_size, int EMPTY = -1, int DELETION = -2>
struct FixHash {
    T data[max_size];

    const uint64_t random_number[16] = {
        0x4bcb391f924ed183ULL, 0xa0ab69ccd854fc0aULL, 0x91086b9cecf5e3b7ULL,
        0xc68e01641bead407ULL, 0x3a7b976128a30449ULL, 0x6d122efabfc4d99fULL,
        0xe6700ef8715030e2ULL, 0x80dd0c3bffcfb45bULL, 0xe80f45af6e4ce166ULL,
        0x6cf43e5aeb53c362ULL, 0x31a27265a93c4f40ULL, 0x743de943cecde0a4ULL,
        0x5ed25dba0288592dULL, 0xa69eb51a362c37bcULL, 0x9a558fed9d4824f0ULL,
        0xf75678c2fdbdd68bULL};

    __device__ FixHash() {
        for (int i = 0; i < max_size; ++i) {
            data[i] = EMPTY;
        }
    }

    __device__ short hash(int h, idx_t x) const {
        return static_cast<short>((x ^ ((x >> 16) * random_number[h << 1]) ^
                                   random_number[(h << 1) + 1]) % max_size);
    }

    __device__ void add(T x) {
        auto code = hash(0, x);
        while (data[code] != EMPTY && data[code] != DELETION) {
            code = (code + 1) % max_size;
        }
        data[code] = x;
    }

    __device__ bool test(T x) const {
        auto code = hash(0, x);
        while (data[code] != EMPTY) {
            if (data[code] == x) return true;
            code = (code + 1) % max_size;
        }
        return false;
    }

    __device__ void del(T x) {
        auto code = hash(0, x);
        auto remove_idx = code;
        while (data[remove_idx] != EMPTY) {
            if (data[remove_idx] == x) break;
            remove_idx = (remove_idx + 1) % max_size;
        }
        if (data[remove_idx] == EMPTY) return;

        int next_idx = (remove_idx + 1) % max_size;
        while (data[next_idx] != EMPTY) {
            auto new_code = hash(0, data[next_idx]);
            bool cond1 = code <= remove_idx;
            bool cond2 = remove_idx < next_idx;
            if (cond1 && cond2) {
                if (new_code <= remove_idx) {
                    data[remove_idx] = data[next_idx];
                    remove_idx = next_idx;
                    code = new_code;
                }
            } else if (cond1) {
                if (next_idx < new_code && new_code <= remove_idx) {
                    data[remove_idx] = data[next_idx];
                    remove_idx = next_idx;
                    code = new_code;
                }
            } else {
                if (next_idx < new_code || new_code <= remove_idx) {
                    data[remove_idx] = data[next_idx];
                    remove_idx = next_idx;
                    code = new_code;
                }
            }
            next_idx = (next_idx + 1) % max_size;
        }
        data[remove_idx] = EMPTY;
    }
};

} // namespace song_kernel

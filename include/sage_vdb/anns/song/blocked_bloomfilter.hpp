#pragma once

#include <cstdint>

#include "sage_vdb/anns/song/config.hpp"

namespace song_kernel {

constexpr int GPU_CACHE_LINE_SIZE64 = 1;
constexpr int GPU_CACHE_LINE_SHIFT = 0;
constexpr int BLOOMFILTER_SIZE64MULT = 2;
constexpr int BLOOMFILTER_SIZE_SHIFT = 5;

template <int size64, int shift, int num_hash>
struct BlockedBloomFilter {
    uint32_t data[size64 * BLOOMFILTER_SIZE64MULT];

    const uint64_t random_number[20] = {
        0x4bcb391f924ed183ULL, 0xa0ab69ccd854fc0aULL, 0x91086b9cecf5e3b7ULL,
        0xc68e01641bead407ULL, 0x3a7b976128a30449ULL, 0x6d122efabfc4d99fULL,
        0xe6700ef8715030e2ULL, 0x80dd0c3bffcfb45bULL, 0xe80f45af6e4ce166ULL,
        0x6cf43e5aeb53c362ULL, 0x31a27265a93c4f40ULL, 0x743de943cecde0a4ULL,
        0x5ed25dba0288592dULL, 0xa69eb51a362c37bcULL, 0x9a558fed9d4824f0ULL,
        0xf75678c2fdbdd68bULL, 0x34423f0963258c85ULL, 0x3532778d6726905cULL,
        0x6fef7cbe609500f9ULL, 0x0b4419d54de48422ULL};

    __device__ BlockedBloomFilter() {
        for (int i = 0; i < size64 * BLOOMFILTER_SIZE64MULT; ++i) data[i] = 0;
    }

    __device__ int pure_hash(int h, idx_t x) const {
        idx_t val = x;
        val ^= val >> 33;
        val *= random_number[h << 1];
        val ^= val >> 33;
        val *= random_number[(h << 1) + 1];
        val ^= val >> 33;
        return static_cast<int>(val);
    }

    __device__ int hash(int h, idx_t x) const {
        idx_t val = x;
        val ^= val >> 33;
        val *= random_number[h << 1];
        val ^= val >> 33;
        val *= random_number[(h << 1) + 1];
        val ^= val >> 33;
        return static_cast<int>(val & ((GPU_CACHE_LINE_SIZE64 << BLOOMFILTER_SIZE_SHIFT) - 1));
    }

    __device__ void set_bit(int offset, int x) {
        data[offset + (x & (GPU_CACHE_LINE_SIZE64 - 1))] |=
            static_cast<uint32_t>(1ULL << (x >> GPU_CACHE_LINE_SHIFT));
    }

    __device__ bool test_bit(int offset, int x) const {
        return (data[offset + (x & (GPU_CACHE_LINE_SIZE64 - 1))] >>
                (x >> GPU_CACHE_LINE_SHIFT)) & 1U;
    }

    __device__ int get_offset(idx_t x) const {
        return (pure_hash(9, x) & ((size64 >> GPU_CACHE_LINE_SHIFT) - 1)) *
               GPU_CACHE_LINE_SIZE64;
    }

    __device__ void add(idx_t x) {
        int offset = get_offset(x);
        for (int i = 0; i < num_hash; ++i) {
            set_bit(offset, hash(i, x));
        }
    }

    __device__ bool test(idx_t x) const {
        int offset = get_offset(x);
        for (int i = 0; i < num_hash; ++i) {
            if (!test_bit(offset, hash(i, x))) return false;
        }
        return true;
    }
};

} // namespace song_kernel

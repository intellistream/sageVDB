#pragma once

#include <cstdint>

#include "sage_vdb/anns/song/config.hpp"

namespace song_kernel {

constexpr int CUCKOO_BUCKET_SIZE = 4;
using cuckoo_bucket_t = uint8_t;

template <int capacity>
class CuckooFilter {
private:
    cuckoo_bucket_t buckets[capacity / CUCKOO_BUCKET_SIZE][CUCKOO_BUCKET_SIZE]{};
    int count = 0;
    const int max_cuckoo_count = capacity / CUCKOO_BUCKET_SIZE;

    __device__ static cuckoo_bucket_t hash_to_fp(idx_t x) {
        return static_cast<cuckoo_bucket_t>((x % 255) + 1);
    }

    __device__ static int fingerprint_index(const cuckoo_bucket_t* bucket, cuckoo_bucket_t fp) {
        for (int i = 0; i < CUCKOO_BUCKET_SIZE; ++i) {
            if (bucket[i] == fp) return i;
        }
        return -1;
    }

    __device__ static bool bucket_insert(cuckoo_bucket_t* bucket, cuckoo_bucket_t fp) {
        for (int i = 0; i < CUCKOO_BUCKET_SIZE; ++i) {
            if (bucket[i] == 0) {
                bucket[i] = fp;
                return true;
            }
        }
        return false;
    }

    __device__ static bool bucket_delete(cuckoo_bucket_t* bucket, cuckoo_bucket_t fp) {
        for (int i = 0; i < CUCKOO_BUCKET_SIZE; ++i) {
            if (bucket[i] == fp) {
                bucket[i] = 0;
                return true;
            }
        }
        return false;
    }

    __device__ static uint32_t hash(idx_t x) {
        uint64_t val = x;
        val ^= val >> 33;
        val *= 0x2b391d3377c181eaULL;
        val ^= val >> 33;
        val *= 0x41e2b6d7fd610dd8ULL;
        val ^= val >> 33;
        return static_cast<uint32_t>(val);
    }

    __device__ int alt_index(cuckoo_bucket_t fp, int idx) const {
        uint32_t h = hash(fp);
        return (idx ^ h) % (capacity / CUCKOO_BUCKET_SIZE);
    }

    __device__ void get_indices_and_fp(idx_t x, int& i1, int& i2, cuckoo_bucket_t& fp) const {
        uint32_t h = hash(x);
        fp = hash_to_fp(x);
        i1 = h % (capacity / CUCKOO_BUCKET_SIZE);
        i2 = alt_index(fp, i1);
    }

    __device__ bool reinsert(cuckoo_bucket_t fp, int idx) {
        for (int k = 0; k < max_cuckoo_count; ++k) {
            int slot = hash(fp + k * 156722 + 1034311351) % CUCKOO_BUCKET_SIZE;
            auto old_fp = fp;
            fp = buckets[idx][slot];
            buckets[idx][slot] = old_fp;
            idx = alt_index(fp, idx);
            if (insert_into_bucket(fp, idx)) return true;
        }
        return false;
    }

    __device__ bool insert_into_bucket(cuckoo_bucket_t fp, int idx) {
        if (bucket_insert(buckets[idx], fp)) {
            ++count;
            return true;
        }
        return false;
    }

    __device__ bool erase_from_bucket(cuckoo_bucket_t fp, int idx) {
        if (bucket_delete(buckets[idx], fp)) {
            --count;
            return true;
        }
        return false;
    }

public:
    __device__ bool test(idx_t x) const {
        int i1, i2;
        cuckoo_bucket_t fp;
        get_indices_and_fp(x, i1, i2, fp);
        return fingerprint_index(buckets[i1], fp) >= 0 ||
               fingerprint_index(buckets[i2], fp) >= 0;
    }

    __device__ bool add(idx_t x) {
        if (test(x)) return false;
        int i1, i2;
        cuckoo_bucket_t fp;
        get_indices_and_fp(x, i1, i2, fp);
        if (insert_into_bucket(fp, i1) || insert_into_bucket(fp, i2)) return true;
        int swap_idx = ((i1 * (i2 >> 5)) ^ i2 ^ buckets[(i1 + i2) / 2][0]) % 2 == 0 ? i1 : i2;
        return reinsert(fp, swap_idx);
    }

    __device__ bool del(idx_t x) {
        int i1, i2;
        cuckoo_bucket_t fp;
        get_indices_and_fp(x, i1, i2, fp);
        return erase_from_bucket(fp, i1) || erase_from_bucket(fp, i2);
    }
};

} // namespace song_kernel

#pragma once

#include "sage_vdb/anns/song/config.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdio>
#include <cstring>
#include <memory>
#include <utility>
#include <vector>

namespace song_kernel {

class Data {
private:
    std::unique_ptr<value_t[]> data_;
    size_t num_;
    size_t curr_num_ = 0;
    int dim_;

public:
    Data(size_t num, int dim) : num_(num), dim_(dim) {
        data_ = std::make_unique<value_t[]>(num_ * dim_);
        std::memset(data_.get(), 0, sizeof(value_t) * num_ * dim_);
    }

    value_t* get(idx_t idx) const { return data_.get() + idx * dim_; }

    void del(idx_t idx) { std::memset(get(idx), 0, sizeof(value_t) * dim_); }

    template <class T>
    dist_t l2_distance(idx_t a, T& v) const {
        auto pa = get(a);
        dist_t ret = 0;
        for (int i = 0; i < dim_; ++i) {
            auto diff = *(pa + i) - v[i];
            ret += diff * diff;
        }
        return ret;
    }

    template <class T>
    dist_t negative_inner_prod_distance(idx_t a, T& v) const {
        auto pa = get(a);
        dist_t ret = 0;
        for (int i = 0; i < dim_; ++i) {
            ret -= (*(pa + i)) * v[i];
        }
        return ret;
    }

    template <class T>
    dist_t negative_cosine_distance(idx_t a, T& v) const {
        auto pa = get(a);
        dist_t ret = 0;
        value_t lena = 0, lenv = 0;
        for (int i = 0; i < dim_; ++i) {
            ret += (*(pa + i)) * v[i];
            lena += (*(pa + i)) * (*(pa + i));
            lenv += v[i] * v[i];
        }
        int sign = ret < 0 ? 1 : -1;
        return sign * (ret * ret / lena / lenv);
    }

    template <class T>
    dist_t real_nn(T& v) const {
        dist_t minn = 1e100;
        for (size_t i = 0; i < curr_num_; ++i) {
            auto res = l2_distance(i, v);
            if (res < minn) {
                minn = res;
            }
        }
        return minn;
    }

    std::vector<value_t> organize_point(const std::vector<std::pair<int, value_t>>& v) {
        std::vector<value_t> ret(dim_, 0);
        for (const auto& p : v) {
            if (p.first >= dim_) {
                std::printf("[SONG] organize_point error: %d %d\n", p.first, dim_);
            }
            ret[p.first] = p.second;
        }
        return ret;
    }

    value_t vec_sum2(const std::vector<std::pair<int, value_t>>& v) {
        value_t ret = 0;
        for (const auto& p : v) {
            if (p.first >= dim_) {
                std::printf("[SONG] vec_sum2 error: %d %d\n", p.first, dim_);
            }
            ret += p.second * p.second;
        }
        return ret;
    }

    void add(idx_t idx, std::vector<std::pair<int, value_t>>& value) {
        curr_num_ = std::max(curr_num_, idx + 1);
        auto p = get(idx);
        for (const auto& v : value) {
            *(p + v.first) = v.second;
        }
    }

    size_t max_vertices() const { return num_; }

    size_t curr_vertices() const { return curr_num_; }

    void dump(const std::string& file = "song.data") const;
    void load(const std::string& file = "song.data");

    int get_dim() const { return dim_; }
};

inline void Data::dump(const std::string& file) const {
    FILE* fp = std::fopen(file.c_str(), "wb");
    if (!fp) {
        return;
    }
    std::fwrite(data_.get(), sizeof(value_t) * num_ * dim_, 1, fp);
    std::fclose(fp);
}

inline void Data::load(const std::string& file) {
    FILE* fp = std::fopen(file.c_str(), "rb");
    if (!fp) {
        return;
    }
    std::fread(data_.get(), sizeof(value_t) * num_ * dim_, 1, fp);
    std::fclose(fp);
    curr_num_ = num_;
}

// Explicit specializations

template <>
inline dist_t Data::l2_distance(idx_t a, idx_t& b) const {
    auto pa = get(a), pb = get(b);
    dist_t ret = 0;
    for (int i = 0; i < dim_; ++i) {
        auto diff = *(pa + i) - *(pb + i);
        ret += diff * diff;
    }
    return ret;
}

template <>
inline dist_t Data::negative_inner_prod_distance(idx_t a, idx_t& b) const {
    auto pa = get(a), pb = get(b);
    dist_t ret = 0;
    for (int i = 0; i < dim_; ++i) {
        ret -= (*(pa + i)) * (*(pb + i));
    }
    return ret;
}

template <>
inline dist_t Data::negative_cosine_distance(idx_t a, idx_t& b) const {
    auto pa = get(a), pb = get(b);
    dist_t ret = 0;
    value_t lena = 0, lenv = 0;
    for (int i = 0; i < dim_; ++i) {
        ret += (*(pa + i)) * (*(pb + i));
        lena += (*(pa + i)) * (*(pa + i));
        lenv += (*(pb + i)) * (*(pb + i));
    }
    int sign = ret < 0 ? 1 : -1;
    return sign * (ret * ret / lena / lenv);
}

} // namespace song_kernel

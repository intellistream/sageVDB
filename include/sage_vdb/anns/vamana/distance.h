#pragma once

#include "sage_vdb/common.h"

#include <cmath>
#include <stdexcept>
#include <vector>

namespace sage_vdb {
namespace anns {
namespace vamana {

class Distance {
public:
    static float l2(const Vector& a, const Vector& b) {
        if (a.size() != b.size()) {
            throw std::invalid_argument("Vamana distance: vector dimensions mismatch");
        }
        float sum = 0.0f;
        for (size_t i = 0; i < a.size(); ++i) {
            float diff = a[i] - b[i];
            sum += diff * diff;
        }
        return std::sqrt(sum);
    }

    static float inner_product(const Vector& a, const Vector& b) {
        if (a.size() != b.size()) {
            throw std::invalid_argument("Vamana distance: vector dimensions mismatch");
        }
        float dot = 0.0f;
        for (size_t i = 0; i < a.size(); ++i) {
            dot += a[i] * b[i];
        }
        return 1.0f - dot;
    }

    static float cosine(const Vector& a, const Vector& b) {
        if (a.size() != b.size()) {
            throw std::invalid_argument("Vamana distance: vector dimensions mismatch");
        }
        float dot = 0.0f;
        float norm_a = 0.0f;
        float norm_b = 0.0f;
        for (size_t i = 0; i < a.size(); ++i) {
            dot += a[i] * b[i];
            norm_a += a[i] * a[i];
            norm_b += b[i] * b[i];
        }
        if (norm_a == 0.0f || norm_b == 0.0f) {
            return 1.0f;
        }
        float cosine_similarity = dot / (std::sqrt(norm_a) * std::sqrt(norm_b));
        return 1.0f - cosine_similarity;
    }
};

} // namespace vamana
} // namespace anns
} // namespace sage_vdb

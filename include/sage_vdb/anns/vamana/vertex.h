#pragma once

#include "sage_vdb/common.h"

#include <vector>

namespace sage_vdb {
namespace anns {
namespace vamana {

using idx_t = uint32_t;

/**
 * @brief Vertex node within the Vamana proximity graph.
 */
class Vertex {
public:
    Vertex() = default;

    Vertex(idx_t identifier, Vector data)
        : id(identifier), vector(std::move(data)) {}

    idx_t id = 0;
    Vector vector;
    std::vector<idx_t> neighbors;
};

} // namespace vamana
} // namespace anns
} // namespace sage_vdb

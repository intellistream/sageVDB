#pragma once

#ifdef ENABLE_FLATGPU_CUDA
#include <cuda_runtime.h>
#endif

#include "sage_vdb/common.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <vector>

namespace sage_vdb {
namespace anns {
namespace flat_gpu {

struct DeviceStats {
    double upload_ms = 0.0;
    double download_ms = 0.0;
    double compute_ms = 0.0;
};

struct QueryLaunchConfig {
    std::size_t batch_size = 1;
    std::size_t top_k = 10;
};

struct DeviceContextDeleter {
    void operator()(void* ptr) noexcept;
};

using DeviceBufferPtr = std::unique_ptr<float, DeviceContextDeleter>;
using DeviceIdBufferPtr = std::unique_ptr<VectorId, DeviceContextDeleter>;
using DeviceFloatBufferPtr = std::unique_ptr<float, DeviceContextDeleter>;

struct DeviceBuffers {
    DeviceBufferPtr vectors;
    DeviceIdBufferPtr ids;
    DeviceFloatBufferPtr norms;
    std::size_t capacity = 0;
    std::size_t count = 0;
    uint32_t dimension = 0;
    int device = -1;
};

struct QueryScratch {
    DeviceBufferPtr queries;
    DeviceFloatBufferPtr query_norms;
    DeviceIdBufferPtr results_ids;
    DeviceBufferPtr results_distances;
    std::size_t query_capacity = 0;
    std::size_t query_norm_capacity = 0;
    std::size_t result_capacity = 0;
};

class CUDABackend {
public:
    static bool is_available();

    static std::unique_ptr<CUDABackend> create(int device);

    ~CUDABackend();

    void upload(DeviceBuffers& buffers,
                const std::vector<float>& host_vectors,
                const std::vector<VectorId>& host_ids,
                DeviceStats& stats);

    void ensure_query_capacity(const DeviceBuffers& buffers,
                               QueryScratch& scratch,
                               std::size_t batch,
                               std::size_t k,
                               uint32_t dimension,
                               DeviceStats& stats);

    void run_query(const DeviceBuffers& buffers,
                   const QueryScratch& scratch,
                   const std::vector<VectorId>& dataset_ids,
                   const float* host_queries,
                   std::size_t batch,
                   std::size_t k,
                   DistanceMetric metric,
                   bool return_distances,
                   float* host_distances,
                   VectorId* host_ids,
                   DeviceStats& stats) const;

    float* query_device_ptr(const QueryScratch& scratch) const;

private:
    explicit CUDABackend(int device);

    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace flat_gpu
} // namespace anns
} // namespace sage_vdb


#include "sage_vdb/anns/flat_gpu/cuda_helpers.h"

#include <stdexcept>
#include <vector>

namespace sage_vdb {
namespace anns {
namespace flat_gpu {

void DeviceContextDeleter::operator()(void* /*ptr*/) noexcept {}

struct CUDABackend::Impl {
};

bool CUDABackend::is_available() {
    return false;
}

std::unique_ptr<CUDABackend> CUDABackend::create(int /*device*/) {
    return nullptr;
}

CUDABackend::~CUDABackend() = default;

void CUDABackend::upload(DeviceBuffers& /*buffers*/,
                         const std::vector<float>& /*host_vectors*/,
                         const std::vector<VectorId>& /*host_ids*/,
                         DeviceStats& /*stats*/) {
    throw std::runtime_error("FlatGPU CUDA backend not available");
}

void CUDABackend::ensure_query_capacity(const DeviceBuffers& /*buffers*/,
                                        QueryScratch& /*scratch*/,
                                        std::size_t /*batch*/,
                                        std::size_t /*k*/,
                                        uint32_t /*dimension*/,
                                        DeviceStats& /*stats*/) {
    throw std::runtime_error("FlatGPU CUDA backend not available");
}

void CUDABackend::run_query(const DeviceBuffers& /*buffers*/,
                            const QueryScratch& /*scratch*/,
                            const std::vector<VectorId>& /*dataset_ids*/,
                            const float* /*host_queries*/,
                            std::size_t /*batch*/,
                            std::size_t /*k*/,
                            DistanceMetric /*metric*/,
                            bool /*return_distances*/,
                            float* /*host_distances*/,
                            VectorId* /*host_ids*/,
                            DeviceStats& /*stats*/) const {
    throw std::runtime_error("FlatGPU CUDA backend not available");
}

float* CUDABackend::query_device_ptr(const QueryScratch& /*scratch*/) const {
    return nullptr;
}

CUDABackend::CUDABackend(int /*device*/) : impl_(nullptr) {}

} // namespace flat_gpu
} // namespace anns
} // namespace sage_vdb

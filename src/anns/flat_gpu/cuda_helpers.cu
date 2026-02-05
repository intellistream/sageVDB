#include "sage_vdb/anns/flat_gpu/cuda_helpers.h"

#ifdef ENABLE_FLATGPU_CUDA

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <mutex>
#include <numeric>
#include <stdexcept>
#include <vector>

namespace sage_vdb {
namespace anns {
namespace flat_gpu {

namespace {

#define CUDA_CHECK(expr)                                                   \
    do {                                                                   \
        cudaError_t _err = (expr);                                         \
        if (_err != cudaSuccess) {                                         \
            throw std::runtime_error(std::string("FlatGPU CUDA error: ") + \
                                     cudaGetErrorString(_err));            \
        }                                                                  \
    } while (false)

constexpr int kBlockSize = 256;

__global__ void compute_distances_kernel(const float* __restrict__ dataset,
                                         const float* __restrict__ dataset_norms,
                                         const float* __restrict__ queries,
                                         const float* __restrict__ query_norms,
                                         float* __restrict__ out_distances,
                                         std::size_t count,
                                         uint32_t dim,
                                         std::size_t batch,
                                         int metric_type) {
    std::size_t data_idx = blockIdx.x * blockDim.x + threadIdx.x;
    std::size_t query_idx = blockIdx.y;

    if (data_idx >= count || query_idx >= batch) {
        return;
    }

    const float* data_vec = dataset + data_idx * dim;
    const float* query_vec = queries + query_idx * dim;

    float result = 0.0f;
    if (metric_type == 0) { // L2
        float accum = 0.0f;
        for (uint32_t i = 0; i < dim; ++i) {
            float diff = data_vec[i] - query_vec[i];
            accum += diff * diff;
        }
        result = sqrtf(accum);
    } else if (metric_type == 1) { // INNER_PRODUCT
        float accum = 0.0f;
        for (uint32_t i = 0; i < dim; ++i) {
            accum += data_vec[i] * query_vec[i];
        }
        result = accum;
    } else { // COSINE
        float dot = 0.0f;
        for (uint32_t i = 0; i < dim; ++i) {
            dot += data_vec[i] * query_vec[i];
        }
        float denom = dataset_norms[data_idx] * query_norms[query_idx];
        if (denom == 0.0f) {
            result = 1.0f;
        } else {
            float cosine = dot / denom;
            cosine = fminf(fmaxf(cosine, -1.0f), 1.0f);
            result = 1.0f - cosine;
        }
    }

    out_distances[query_idx * count + data_idx] = result;
}

inline int metric_to_int(DistanceMetric metric) {
    switch (metric) {
        case DistanceMetric::L2:
            return 0;
        case DistanceMetric::INNER_PRODUCT:
            return 1;
        case DistanceMetric::COSINE:
            return 2;
    }
    return 0;
}

} // namespace

void DeviceContextDeleter::operator()(void* ptr) noexcept {
    if (ptr != nullptr) {
        cudaFree(ptr);
    }
}

struct CUDABackend::Impl {
    int device = -1;
    cudaStream_t stream = nullptr;
    std::vector<VectorId> host_ids;
};

bool CUDABackend::is_available() {
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess) {
        return false;
    }
    return count > 0;
}

std::unique_ptr<CUDABackend> CUDABackend::create(int device) {
    if (!is_available()) {
        return nullptr;
    }

    if (device < 0) {
        CUDA_CHECK(cudaGetDevice(&device));
    }

    CUDA_CHECK(cudaSetDevice(device));
    return std::unique_ptr<CUDABackend>(new CUDABackend(device));
}

CUDABackend::CUDABackend(int device)
    : impl_(std::make_unique<Impl>()) {
    impl_->device = device;
    CUDA_CHECK(cudaSetDevice(device));
    CUDA_CHECK(cudaStreamCreateWithFlags(&impl_->stream, cudaStreamNonBlocking));
}

CUDABackend::~CUDABackend() {
    if (impl_) {
        cudaSetDevice(impl_->device);
        if (impl_->stream != nullptr) {
            cudaStreamDestroy(impl_->stream);
        }
    }
}

void CUDABackend::upload(DeviceBuffers& buffers,
                         const std::vector<float>& host_vectors,
                         const std::vector<VectorId>& host_ids,
                         DeviceStats& stats) {
    CUDA_CHECK(cudaSetDevice(impl_->device));

    const std::size_t count = host_ids.size();
    if (count == 0) {
        buffers.count = 0;
        impl_->host_ids.clear();
        return;
    }

    if (buffers.device != impl_->device) {
        buffers.device = impl_->device;
    }

    const uint32_t dimension = buffers.dimension;
    if (dimension == 0) {
        throw std::runtime_error("FlatGPU CUDA upload: dimension not set");
    }

    if (buffers.capacity < count) {
        std::size_t new_capacity = std::max<std::size_t>(count, buffers.capacity * 2);

        float* new_vectors = nullptr;
        CUDA_CHECK(cudaMalloc(&new_vectors, new_capacity * dimension * sizeof(float)));
        buffers.vectors.reset(new_vectors);

        VectorId* new_ids = nullptr;
        CUDA_CHECK(cudaMalloc(&new_ids, new_capacity * sizeof(VectorId)));
        buffers.ids.reset(new_ids);

        float* new_norms = nullptr;
        CUDA_CHECK(cudaMalloc(&new_norms, new_capacity * sizeof(float)));
        buffers.norms.reset(new_norms);

        buffers.capacity = new_capacity;
    }

    std::vector<float> norms(count, 0.0f);
    for (std::size_t i = 0; i < count; ++i) {
        const float* row = host_vectors.data() + i * dimension;
        float accum = 0.0f;
        for (uint32_t d = 0; d < dimension; ++d) {
            accum += row[d] * row[d];
        }
        norms[i] = std::sqrt(accum);
    }

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start, impl_->stream));

    CUDA_CHECK(cudaMemcpyAsync(buffers.vectors.get(),
                               host_vectors.data(),
                               count * dimension * sizeof(float),
                               cudaMemcpyHostToDevice,
                               impl_->stream));
    CUDA_CHECK(cudaMemcpyAsync(buffers.ids.get(),
                               host_ids.data(),
                               count * sizeof(VectorId),
                               cudaMemcpyHostToDevice,
                               impl_->stream));
    CUDA_CHECK(cudaMemcpyAsync(buffers.norms.get(),
                               norms.data(),
                               count * sizeof(float),
                               cudaMemcpyHostToDevice,
                               impl_->stream));

    CUDA_CHECK(cudaEventRecord(stop, impl_->stream));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    stats.upload_ms = ms;

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    buffers.count = count;
    impl_->host_ids = host_ids;
}

void CUDABackend::ensure_query_capacity(const DeviceBuffers& buffers,
                                        QueryScratch& scratch,
                                        std::size_t batch,
                                        std::size_t /*k*/,
                                        uint32_t dimension,
                                        DeviceStats& /*stats*/) {
    CUDA_CHECK(cudaSetDevice(impl_->device));

    std::size_t needed_query_elems = batch * dimension;
    if (scratch.query_capacity < needed_query_elems) {
        float* device_queries = nullptr;
        CUDA_CHECK(cudaMalloc(&device_queries, needed_query_elems * sizeof(float)));
        scratch.queries.reset(device_queries);
        scratch.query_capacity = needed_query_elems;
    }

    if (!scratch.query_norms || scratch.query_norm_capacity < batch) {
        float* device_query_norms = nullptr;
        CUDA_CHECK(cudaMalloc(&device_query_norms, batch * sizeof(float)));
        scratch.query_norms.reset(device_query_norms);
        scratch.query_norm_capacity = batch;
    }

    if (!scratch.results_distances || scratch.result_capacity < buffers.count) {
        float* device_distances = nullptr;
        CUDA_CHECK(cudaMalloc(&device_distances, batch * buffers.count * sizeof(float)));
        scratch.results_distances.reset(device_distances);
        scratch.result_capacity = buffers.count;
    }

    if (!scratch.results_ids) {
        VectorId* device_ids = nullptr;
        CUDA_CHECK(cudaMalloc(&device_ids, buffers.count * sizeof(VectorId)));
        scratch.results_ids.reset(device_ids);
    }
}

void CUDABackend::run_query(const DeviceBuffers& buffers,
                            const QueryScratch& scratch,
                            const std::vector<VectorId>& dataset_ids,
                            const float* host_queries,
                            std::size_t batch,
                            std::size_t k,
                            DistanceMetric metric,
                            bool return_distances,
                            float* host_distances,
                            VectorId* host_ids,
                            DeviceStats& stats) const {
    if (buffers.count == 0 || batch == 0) {
        return;
    }

    CUDA_CHECK(cudaSetDevice(impl_->device));

    std::vector<float> query_norms(batch, 0.0f);
    for (std::size_t q = 0; q < batch; ++q) {
        const float* query_vec = host_queries + q * buffers.dimension;
        float accum = 0.0f;
        for (uint32_t d = 0; d < buffers.dimension; ++d) {
            accum += query_vec[d] * query_vec[d];
        }
        query_norms[q] = std::sqrt(accum);
    }

    cudaEvent_t compute_start, compute_stop;
    CUDA_CHECK(cudaEventCreate(&compute_start));
    CUDA_CHECK(cudaEventCreate(&compute_stop));

    CUDA_CHECK(cudaMemcpyAsync(scratch.queries.get(),
                               host_queries,
                               batch * buffers.dimension * sizeof(float),
                               cudaMemcpyHostToDevice,
                               impl_->stream));
    CUDA_CHECK(cudaMemcpyAsync(scratch.query_norms.get(),
                               query_norms.data(),
                               batch * sizeof(float),
                               cudaMemcpyHostToDevice,
                               impl_->stream));

    CUDA_CHECK(cudaEventRecord(compute_start, impl_->stream));

    dim3 block(kBlockSize);
    dim3 grid((buffers.count + block.x - 1) / block.x, batch);
    compute_distances_kernel<<<grid, block, 0, impl_->stream>>>(
        buffers.vectors.get(),
        buffers.norms.get(),
        scratch.queries.get(),
        scratch.query_norms.get(),
        scratch.results_distances.get(),
        buffers.count,
        buffers.dimension,
        batch,
        metric_to_int(metric));

    CUDA_CHECK(cudaEventRecord(compute_stop, impl_->stream));
    CUDA_CHECK(cudaEventSynchronize(compute_stop));

    float compute_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&compute_ms, compute_start, compute_stop));
    stats.compute_ms = compute_ms;

    CUDA_CHECK(cudaEventDestroy(compute_start));
    CUDA_CHECK(cudaEventDestroy(compute_stop));

    std::vector<float> host_distance_buffer(batch * buffers.count);
    auto download_start = std::chrono::high_resolution_clock::now();
    CUDA_CHECK(cudaMemcpy(host_distance_buffer.data(),
                          scratch.results_distances.get(),
                          batch * buffers.count * sizeof(float),
                          cudaMemcpyDeviceToHost));
    auto download_end = std::chrono::high_resolution_clock::now();
    stats.download_ms = std::chrono::duration<double, std::milli>(download_end - download_start).count();

    for (std::size_t q = 0; q < batch; ++q) {
        std::vector<std::pair<float, VectorId>> scored;
        scored.reserve(buffers.count);

        for (std::size_t i = 0; i < buffers.count; ++i) {
            float dist = host_distance_buffer[q * buffers.count + i];
            scored.emplace_back(dist, dataset_ids[i]);
        }

        auto comparator = [metric](const auto& a, const auto& b) {
            if (metric == DistanceMetric::INNER_PRODUCT) {
                return a.first > b.first;
            }
            return a.first < b.first;
        };

        std::size_t actual_k = std::min(k, buffers.count);
        std::partial_sort(scored.begin(), scored.begin() + actual_k, scored.end(), comparator);

        for (std::size_t i = 0; i < actual_k; ++i) {
            host_ids[q * k + i] = scored[i].second;
            if (return_distances && host_distances != nullptr) {
                host_distances[q * k + i] = scored[i].first;
            }
        }

        for (std::size_t i = actual_k; i < k; ++i) {
            host_ids[q * k + i] = 0;
            if (return_distances && host_distances != nullptr) {
                if (metric == DistanceMetric::INNER_PRODUCT) {
                    host_distances[q * k + i] = -std::numeric_limits<float>::infinity();
                } else {
                    host_distances[q * k + i] = std::numeric_limits<float>::infinity();
                }
            }
        }
    }
}

float* CUDABackend::query_device_ptr(const QueryScratch& scratch) const {
    return scratch.queries.get();
}

} // namespace flat_gpu
} // namespace anns
} // namespace sage_vdb

#endif // ENABLE_FLATGPU_CUDA

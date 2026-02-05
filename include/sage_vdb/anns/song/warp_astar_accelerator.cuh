#pragma once

#include <cuda.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>

#include "sage_vdb/anns/song/bin_heap.hpp"
#include "sage_vdb/anns/song/config.hpp"
#include "sage_vdb/anns/song/data.hpp"
#include "sage_vdb/anns/song/fixhash.hpp"
#include "sage_vdb/anns/song/smmh2.hpp"

namespace song_kernel {

constexpr unsigned FULL_MASK = 0xffffffffu;
constexpr int N_THREAD_IN_WARP = 32;
constexpr int N_MULTIQUERY = 1;
constexpr int CRITICAL_STEP = N_THREAD_IN_WARP / N_MULTIQUERY;
constexpr int N_MULTIPROBE = 1;
constexpr int FINISH_CNT = 1;
constexpr int BLOOM_FILTER_BIT64 = 8;
constexpr int BLOOM_FILTER_BIT_SHIFT = 3;
constexpr int BLOOM_FILTER_NUM_HASH = 7;
constexpr int HASH_TABLE_CAPACITY = 10 * 4 * 2;

struct Measure {
    unsigned long long stage1 = 0;
    unsigned long long stage2 = 0;
    unsigned long long stage3 = 0;
};

template <class A, class B>
struct KernelPair {
    A first;
    B second;

    __device__ KernelPair() = default;

    __device__ bool operator<(KernelPair& kp) const { return first < kp.first; }

    __device__ bool operator>(KernelPair& kp) const { return first > kp.first; }
};

__global__ static void warp_independent_search_kernel(
    const value_t* d_data, const value_t* d_query, idx_t* d_result, const idx_t* d_graph, int num,
    int vertex_offset_shift, int annk, Measure* measure, int dist_type, int dim) {
    int QUEUE_SIZE = annk;
    int bid = blockIdx.x * N_MULTIQUERY;
    const int step = N_THREAD_IN_WARP;
    int tid = threadIdx.x;
    int cid = tid / CRITICAL_STEP;
    int subtid = tid % CRITICAL_STEP;

    FixHash<int, HASH_TABLE_CAPACITY>* pbf;
    KernelPair<dist_t, idx_t>* q;
    KernelPair<dist_t, idx_t>* topk;
    value_t* dist_list;
    if (subtid == 0) {
        dist_list = new value_t[FIXED_DEGREE * N_MULTIPROBE];
        q = new KernelPair<dist_t, idx_t>[QUEUE_SIZE + 2];
        topk = new KernelPair<dist_t, idx_t>[annk + 1];
        pbf = new FixHash<int, HASH_TABLE_CAPACITY>();
    }
    __shared__ int heap_size[N_MULTIQUERY];
    int topk_heap_size;

    __shared__ value_t query_point[N_MULTIQUERY][ACC_BATCH_SIZE];
    __shared__ int finished[N_MULTIQUERY];
    __shared__ idx_t index_list[N_MULTIQUERY][FIXED_DEGREE * N_MULTIPROBE];
    __shared__ char index_list_len[N_MULTIQUERY];
    value_t start_distance = 0;
    __syncthreads();

    value_t tmp[N_MULTIQUERY];
    value_t tmp_data_len[N_MULTIQUERY];
    for (int j = 0; j < N_MULTIQUERY; ++j) {
        tmp[j] = 0;
        tmp_data_len[j] = 0;
        for (int i = tid; i < dim; i += step) {
            query_point[j][i] = d_query[(bid + j) * dim + i];
            if (dist_type == 0) {
                value_t diff = query_point[j][i] - d_data[i];
                tmp[j] += diff * diff;
            } else if (dist_type == 1) {
                tmp[j] += query_point[j][i] * d_data[i];
            } else if (dist_type == 2) {
                tmp[j] += query_point[j][i] * d_data[i];
                tmp_data_len[j] += d_data[i] * d_data[i];
            }
        }
        for (int offset = 16; offset > 0; offset /= 2) {
            tmp[j] += __shfl_xor_sync(FULL_MASK, tmp[j], offset);
            if (dist_type == 2) {
                tmp_data_len[j] += __shfl_xor_sync(FULL_MASK, tmp_data_len[j], offset);
            }
        }
    }
    if (subtid == 0) {
        if (dist_type == 0) {
            start_distance = tmp[cid];
        } else if (dist_type == 1) {
            start_distance = -tmp[cid];
        } else if (dist_type == 2) {
            int sign = tmp[cid] < 0 ? 1 : -1;
            if (tmp_data_len[cid] != 0) {
                start_distance =
                    sign * tmp[cid] * tmp[cid] / tmp_data_len[cid];
            } else {
                start_distance = 0;
            }
        }
    }
    __syncthreads();

    if (subtid == 0) {
        heap_size[cid] = 1;
        topk_heap_size = 0;
        finished[cid] = false;
        KernelPair<dist_t, idx_t> kp;
        kp.first = start_distance;
        kp.second = 0;
        smmh2::insert(q, heap_size[cid], kp);
        pbf->add(0);
    }
    __syncthreads();

    while (heap_size[cid] > 1) {
        auto stage1_start = clock64();
        index_list_len[cid] = 0;
        int current_heap_elements = heap_size[cid] - 1;
        for (int k = 0; k < N_MULTIPROBE && k < current_heap_elements; ++k) {
            KernelPair<dist_t, idx_t> now;
            if (subtid == 0) {
                now = smmh2::pop_min(q, heap_size[cid]);
                pbf->del(now.second);
                if (k == 0 && topk_heap_size == annk && topk[0].first <= now.first) {
                    ++finished[cid];
                }
            }
            __syncthreads();
            if (finished[cid] >= FINISH_CNT) break;
            if (subtid == 0) {
                topk[topk_heap_size++] = now;
                push_heap(topk, topk + topk_heap_size);
                pbf->add(now.second);
                if (topk_heap_size > annk) {
                    pbf->del(topk[0].second);
                    pop_heap(topk, topk + topk_heap_size);
                    --topk_heap_size;
                }
                auto offset = now.second << vertex_offset_shift;
                int degree = d_graph[offset];
                for (int i = 1; i <= degree; ++i) {
                    auto idx = d_graph[offset + i];
                    if (pbf->test(idx)) {
                        continue;
                    }
                    index_list[cid][index_list_len[cid]++] = idx;
                }
            }
        }
        if (finished[cid] >= FINISH_CNT) break;
        __syncthreads();

        auto stage1_end = clock64();
        if (tid == 0) atomicAdd(&measure->stage1, stage1_end - stage1_start);
        auto stage2_start = clock64();

        for (int nq = 0; nq < N_MULTIQUERY; ++nq) {
            for (int i = 0; i < index_list_len[nq]; ++i) {
                value_t tmp_val = 0;
                value_t tmp_len = 0;
                for (int j = tid; j < dim; j += step) {
                    if (dist_type == 0) {
                        value_t diff = query_point[nq][j] - d_data[index_list[nq][i] * dim + j];
                        tmp_val += diff * diff;
                    } else if (dist_type == 1) {
                        tmp_val += query_point[nq][j] * d_data[index_list[nq][i] * dim + j];
                    } else if (dist_type == 2) {
                        tmp_val += query_point[nq][j] * d_data[index_list[nq][i] * dim + j];
                        tmp_len +=
                            d_data[index_list[nq][i] * dim + j] * d_data[index_list[nq][i] * dim + j];
                    }
                }
                for (int offset = 16; offset > 0; offset /= 2) {
                    tmp_val += __shfl_xor_sync(FULL_MASK, tmp_val, offset);
                    if (dist_type == 2) {
                        tmp_len += __shfl_xor_sync(FULL_MASK, tmp_len, offset);
                    }
                }
                if (tid == nq * CRITICAL_STEP) {
                    if (dist_type == 0) {
                        dist_list[i] = tmp_val;
                    } else if (dist_type == 1) {
                        dist_list[i] = -tmp_val;
                    } else if (dist_type == 2) {
                        int sign = tmp_val < 0 ? 1 : -1;
                        if (tmp_len != 0) {
                            dist_list[i] = sign * tmp_val * tmp_val / tmp_len;
                        } else {
                            dist_list[i] = 0;
                        }
                    }
                }
            }
        }

        __syncthreads();
        auto stage2_end = clock64();
        if (tid == 0) atomicAdd(&measure->stage2, stage2_end - stage2_start);
        auto stage3_start = clock64();

        if (subtid == 0) {
            for (int i = 0; i < index_list_len[cid]; ++i) {
                KernelPair<dist_t, idx_t> kp;
                kp.first = dist_list[i];
                kp.second = index_list[cid][i];
                if (heap_size[cid] >= QUEUE_SIZE + 1 && q[2].first < kp.first) {
                    continue;
                }
                smmh2::insert(q, heap_size[cid], kp);
                pbf->add(kp.second);
                if (heap_size[cid] >= QUEUE_SIZE + 2) {
                    pbf->del(q[2].second);
                    smmh2::pop_max(q, heap_size[cid]);
                }
            }
        }
        __syncthreads();
        auto stage3_end = clock64();
        if (tid == 0) atomicAdd(&measure->stage3, stage3_end - stage3_start);
    }

    if (subtid == 0) {
        for (int i = 0; i < annk; ++i) {
            auto now = pop_heap(topk, topk + topk_heap_size - i);
            d_result[(bid + cid) * annk + annk - 1 - i] = now.second;
        }
        delete[] q;
        delete[] topk;
        delete pbf;
        delete[] dist_list;
    }
}

class WarpAStarAccelerator {
private:
    static void check_cuda_status(cudaError_t status, const char* msg) {
        if (status != cudaSuccess) {
            throw std::runtime_error(std::string("[SONG] ") + msg + ": " +
                                     cudaGetErrorString(status));
        }
    }

public:
    template <class QueryContainer, class ResultContainer>
    static void astar_multi_start_search_batch(const QueryContainer& queries, int annk,
                                               ResultContainer& results, const value_t* h_data,
                                               const idx_t* h_graph, int vertex_offset_shift,
                                               int num, int dim, int dist_type) {
        if (dim > ACC_BATCH_SIZE) {
            throw std::runtime_error(
                "[SONG] Query dimension exceeds ACC_BATCH_SIZE shared-memory limit");
        }

        value_t* d_data = nullptr;
        value_t* d_query = nullptr;
        idx_t* d_result = nullptr;
        idx_t* d_graph = nullptr;
        Measure* d_measure = nullptr;
        Measure h_measure{};

        check_cuda_status(cudaMalloc(&d_data, sizeof(value_t) * num * dim), "cudaMalloc d_data");
        check_cuda_status(cudaMalloc(&d_graph, sizeof(idx_t) * (num << vertex_offset_shift)),
                          "cudaMalloc d_graph");
        check_cuda_status(cudaMemcpy(d_data, h_data, sizeof(value_t) * num * dim,
                                     cudaMemcpyHostToDevice),
                          "cudaMemcpy d_data");
        check_cuda_status(cudaMemcpy(d_graph, h_graph,
                                     sizeof(idx_t) * (num << vertex_offset_shift),
                                     cudaMemcpyHostToDevice),
                          "cudaMemcpy d_graph");

        check_cuda_status(cudaMalloc(&d_measure, sizeof(Measure)), "cudaMalloc d_measure");
        check_cuda_status(cudaMemcpy(d_measure, &h_measure, sizeof(Measure), cudaMemcpyHostToDevice),
                          "cudaMemcpy d_measure init");

        auto time_begin = std::chrono::steady_clock::now();
        std::unique_ptr<value_t[]> h_query(new value_t[queries.size() * dim]);
        std::memset(h_query.get(), 0, sizeof(value_t) * queries.size() * dim);

        for (size_t i = 0; i < queries.size(); ++i) {
            for (const auto& p : queries[i]) {
                h_query[i * dim + p.first] = p.second;
            }
        }

        std::unique_ptr<idx_t[]> h_result(new idx_t[queries.size() * annk]);

        check_cuda_status(cudaMalloc(&d_query, sizeof(value_t) * queries.size() * dim),
                          "cudaMalloc d_query");
        check_cuda_status(cudaMalloc(&d_result, sizeof(idx_t) * queries.size() * annk),
                          "cudaMalloc d_result");
        check_cuda_status(cudaMemcpy(d_query, h_query.get(), sizeof(value_t) * queries.size() * dim,
                                     cudaMemcpyHostToDevice),
                          "cudaMemcpy d_query");

        dim3 grid(std::max<size_t>(1, queries.size() / N_MULTIQUERY));
        dim3 block(N_THREAD_IN_WARP);
        warp_independent_search_kernel<<<grid, block>>>(d_data, d_query, d_result, d_graph, num,
                                                        vertex_offset_shift, annk, d_measure,
                                                        dist_type, dim);
        check_cuda_status(cudaDeviceSynchronize(), "cudaDeviceSynchronize");

        check_cuda_status(cudaMemcpy(h_result.get(), d_result,
                                     sizeof(idx_t) * queries.size() * annk,
                                     cudaMemcpyDeviceToHost),
                          "cudaMemcpy result");
        check_cuda_status(cudaMemcpy(&h_measure, d_measure, sizeof(Measure), cudaMemcpyDeviceToHost),
                          "cudaMemcpy measure");

        results.clear();
        for (size_t i = 0; i < queries.size(); ++i) {
            std::vector<idx_t> v(annk);
            for (int j = 0; j < annk; ++j) {
                v[j] = h_result[i * annk + j];
            }
            results.push_back(std::move(v));
        }

        auto time_end = std::chrono::steady_clock::now();
        (void)time_begin;
        (void)time_end;

        cudaFree(d_data);
        cudaFree(d_query);
        cudaFree(d_result);
        cudaFree(d_graph);
        cudaFree(d_measure);
    }

    template <class Query, class ResultContainer>
    static void astar_multi_start_search(const Query& query, int annk, ResultContainer& result,
                                         const value_t* h_data, const idx_t* h_graph,
                                         int vertex_offset_shift, int num, int dim, int dist_type) {
        std::vector<std::vector<std::pair<int, value_t>>> batch_queries(1, query);
        std::vector<std::vector<idx_t>> batch_results;
        astar_multi_start_search_batch(batch_queries, annk, batch_results, h_data, h_graph,
                                       vertex_offset_shift, num, dim, dist_type);
        if (!batch_results.empty()) {
            result = std::move(batch_results.front());
        }
    }
};

} // namespace song_kernel

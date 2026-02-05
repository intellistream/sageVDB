#include "sage_vdb/anns/song_plugin.h"

#ifdef ENABLE_SONG

#include "sage_vdb/common.h"
#include "sage_vdb/anns/song/data.hpp"
#include "sage_vdb/anns/song/kernelgraph.cuh"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace sage_vdb {
namespace anns {

namespace {
REGISTER_ANNS_ALGORITHM(SongANNSFactory);
}

class SongANNS::Impl {
public:
    Impl() = default;
    ~Impl() = default;

    void reset() {
        id_map_.clear();
        reverse_id_map_.clear();
        data_.reset();
        graph_.reset();
        next_internal_idx_ = 0;
        built_ = false;
    }

    std::unique_ptr<song_kernel::Data> data_;
    std::unique_ptr<song_kernel::GraphWrapper> graph_;
    
    std::unordered_map<VectorId, song_kernel::idx_t> id_map_;
    std::unordered_map<song_kernel::idx_t, VectorId> reverse_id_map_;
    song_kernel::idx_t next_internal_idx_ = 0;
    bool built_ = false;
};

SongANNS::SongANNS()
    : impl_(std::make_unique<Impl>()),
      metric_(DistanceMetric::L2),
      dimension_(0),
      max_vectors_(0),
      is_built_(false) {
    metrics_.reset();
}

SongANNS::~SongANNS() = default;

std::string SongANNS::description() const {
    return "SONG GPU-accelerated graph-based ANN with CUDA warp A* search";
}

std::vector<DistanceMetric> SongANNS::supported_distances() const {
    return {DistanceMetric::L2, DistanceMetric::INNER_PRODUCT, DistanceMetric::COSINE};
}

bool SongANNS::supports_distance(DistanceMetric metric) const {
    const auto supported = supported_distances();
    return std::find(supported.begin(), supported.end(), metric) != supported.end();
}

namespace {
// Helper: convert dense Vector to sparse representation for SONG kernels
std::vector<std::pair<int, song_kernel::value_t>> to_sparse_vector(const Vector& vec) {
    std::vector<std::pair<int, song_kernel::value_t>> sparse;
    sparse.reserve(vec.size());
    for (size_t i = 0; i < vec.size(); ++i) {
        sparse.emplace_back(static_cast<int>(i), static_cast<song_kernel::value_t>(vec[i]));
    }
    return sparse;
}

int distance_metric_to_kernel_type(DistanceMetric metric) {
    switch (metric) {
        case DistanceMetric::L2: return 0;
        case DistanceMetric::INNER_PRODUCT: return 1;
        case DistanceMetric::COSINE: return 2;
        default:
            throw std::runtime_error("SONG: unsupported distance metric");
    }
}
} // namespace

void SongANNS::fit(const std::vector<VectorEntry>& dataset, const AlgorithmParams& params) {
    auto start = std::chrono::high_resolution_clock::now();
    
    impl_->reset();
    metrics_.reset();

    if (dataset.empty()) {
        dimension_ = 0;
        max_vectors_ = 0;
        is_built_ = true;
        impl_->built_ = true;
        return;
    }

    dimension_ = static_cast<int>(dataset.front().second.size());
    for (const auto& entry : dataset) {
        if (entry.second.size() != static_cast<size_t>(dimension_)) {
            throw std::runtime_error("SONG: inconsistent vector dimensions in dataset");
        }
    }

    metric_ = static_cast<DistanceMetric>(
        params.get<int>("metric", static_cast<int>(DistanceMetric::L2)));
    if (!supports_distance(metric_)) {
        throw std::runtime_error("SONG: requested distance metric not supported");
    }

    size_t capacity = params.get<size_t>("capacity", dataset.size() * 2);
    max_vectors_ = std::max(dataset.size(), capacity);

    impl_->data_ = std::make_unique<song_kernel::Data>(max_vectors_, dimension_);
    
    int kernel_dist_type = distance_metric_to_kernel_type(metric_);
    
    switch (kernel_dist_type) {
        case 0:
            impl_->graph_ = std::make_unique<song_kernel::KernelFixedDegreeGraph<0>>(impl_->data_.get());
            break;
        case 1:
            impl_->graph_ = std::make_unique<song_kernel::KernelFixedDegreeGraph<1>>(impl_->data_.get());
            break;
        case 2:
            impl_->graph_ = std::make_unique<song_kernel::KernelFixedDegreeGraph<2>>(impl_->data_.get());
            break;
        default:
            throw std::runtime_error("SONG: invalid distance type");
    }

    for (const auto& entry : dataset) {
        VectorId external_id = entry.first;
        const Vector& vec = entry.second;
        
        song_kernel::idx_t internal_idx = impl_->next_internal_idx_++;
        impl_->id_map_[external_id] = internal_idx;
        impl_->reverse_id_map_[internal_idx] = external_id;
        
        auto sparse_vec = to_sparse_vector(vec);
        impl_->data_->add(internal_idx, sparse_vec);
        impl_->graph_->add_vertex(internal_idx, sparse_vec);
    }

    is_built_ = true;
    impl_->built_ = true;
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    metrics_.build_time_seconds = duration.count() / 1000000.0;
    metrics_.index_size_bytes = get_memory_usage();
}

bool SongANNS::save(const std::string& path) const {
    if (!is_built_ || !impl_->graph_ || !impl_->data_) {
        return false;
    }
    
    try {
        std::string graph_file = path + ".graph";
        std::string data_file = path + ".data";
        std::string meta_file = path + ".meta";
        
        impl_->graph_->dump(graph_file);
        impl_->data_->dump(data_file);
        
        FILE* fp = std::fopen(meta_file.c_str(), "wb");
        if (!fp) return false;
        
        int metric_int = static_cast<int>(metric_);
        std::fwrite(&metric_int, sizeof(int), 1, fp);
        std::fwrite(&dimension_, sizeof(int), 1, fp);
        
        size_t id_map_size = impl_->id_map_.size();
        std::fwrite(&id_map_size, sizeof(size_t), 1, fp);
        for (const auto& [ext_id, int_idx] : impl_->id_map_) {
            std::fwrite(&ext_id, sizeof(VectorId), 1, fp);
            std::fwrite(&int_idx, sizeof(song_kernel::idx_t), 1, fp);
        }
        
        std::fclose(fp);
        return true;
    } catch (...) {
        return false;
    }
}

bool SongANNS::load(const std::string& path) {
    impl_->reset();
    
    try {
        std::string graph_file = path + ".graph";
        std::string data_file = path + ".data";
        std::string meta_file = path + ".meta";
        
        FILE* fp = std::fopen(meta_file.c_str(), "rb");
        if (!fp) return false;
        
        int metric_int;
        std::fread(&metric_int, sizeof(int), 1, fp);
        metric_ = static_cast<DistanceMetric>(metric_int);
        std::fread(&dimension_, sizeof(int), 1, fp);
        
        size_t id_map_size;
        std::fread(&id_map_size, sizeof(size_t), 1, fp);
        
        for (size_t i = 0; i < id_map_size; ++i) {
            VectorId ext_id;
            song_kernel::idx_t int_idx;
            std::fread(&ext_id, sizeof(VectorId), 1, fp);
            std::fread(&int_idx, sizeof(song_kernel::idx_t), 1, fp);
            impl_->id_map_[ext_id] = int_idx;
            impl_->reverse_id_map_[int_idx] = ext_id;
            impl_->next_internal_idx_ = std::max(impl_->next_internal_idx_, int_idx + 1);
        }
        
        std::fclose(fp);
        
        max_vectors_ = impl_->next_internal_idx_ * 2;
        impl_->data_ = std::make_unique<song_kernel::Data>(max_vectors_, dimension_);
        impl_->data_->load(data_file);
        
        int kernel_dist_type = distance_metric_to_kernel_type(metric_);
        switch (kernel_dist_type) {
            case 0:
                impl_->graph_ = std::make_unique<song_kernel::KernelFixedDegreeGraph<0>>(impl_->data_.get());
                break;
            case 1:
                impl_->graph_ = std::make_unique<song_kernel::KernelFixedDegreeGraph<1>>(impl_->data_.get());
                break;
            case 2:
                impl_->graph_ = std::make_unique<song_kernel::KernelFixedDegreeGraph<2>>(impl_->data_.get());
                break;
        }
        
        impl_->graph_->load(graph_file);
        
        is_built_ = true;
        impl_->built_ = true;
        return true;
    } catch (...) {
        impl_->reset();
        is_built_ = false;
        return false;
    }
}

bool SongANNS::is_built() const {
    return is_built_;
}

ANNSResult SongANNS::query(const Vector& query_vector, const QueryConfig& config) const {
    if (!is_built_ || !impl_->graph_) {
        throw std::runtime_error("SONG: index not built");
    }
    
    if (query_vector.size() != static_cast<size_t>(dimension_)) {
        throw std::runtime_error("SONG: query dimension mismatch");
    }
    
    auto start = std::chrono::high_resolution_clock::now();
    
    auto sparse_query = to_sparse_vector(query_vector);
    std::vector<song_kernel::idx_t> internal_results;
    
    impl_->graph_->search_top_k(sparse_query, config.k, internal_results);
    
    ANNSResult result;
    result.ids.reserve(internal_results.size());
    result.distances.reserve(internal_results.size());
    
    for (auto internal_idx : internal_results) {
        auto it = impl_->reverse_id_map_.find(internal_idx);
        if (it != impl_->reverse_id_map_.end()) {
            result.ids.push_back(it->second);
            
            if (config.return_distances) {
                auto sparse_internal = to_sparse_vector(query_vector);
                song_kernel::dist_t dist = 0.0;
                
                switch (distance_metric_to_kernel_type(metric_)) {
                    case 0:
                        dist = impl_->data_->l2_distance(internal_idx, sparse_internal);
                        break;
                    case 1:
                        dist = impl_->data_->negative_inner_prod_distance(internal_idx, sparse_internal);
                        dist = -dist;
                        break;
                    case 2:
                        dist = impl_->data_->negative_cosine_distance(internal_idx, sparse_internal);
                        dist = -dist;
                        break;
                }
                
                result.distances.push_back(static_cast<float>(dist));
            }
        }
    }
    
    result.actual_k = result.ids.size();
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    metrics_.search_time_seconds += duration.count() / 1000000.0;
    metrics_.distance_computations += result.actual_k;
    
    return result;
}

std::vector<ANNSResult> SongANNS::batch_query(const std::vector<Vector>& query_vectors,
                                              const QueryConfig& config) const {
    if (!is_built_ || !impl_->graph_) {
        throw std::runtime_error("SONG: index not built");
    }
    
    auto start = std::chrono::high_resolution_clock::now();
    
    std::vector<std::vector<std::pair<int, song_kernel::value_t>>> sparse_queries;
    sparse_queries.reserve(query_vectors.size());
    
    for (const auto& qv : query_vectors) {
        if (qv.size() != static_cast<size_t>(dimension_)) {
            throw std::runtime_error("SONG: query dimension mismatch");
        }
        sparse_queries.push_back(to_sparse_vector(qv));
    }
    
    std::vector<std::vector<song_kernel::idx_t>> internal_results;
    impl_->graph_->search_top_k_batch(sparse_queries, config.k, internal_results);
    
    std::vector<ANNSResult> results;
    results.reserve(query_vectors.size());
    
    for (size_t i = 0; i < internal_results.size(); ++i) {
        ANNSResult result;
        result.ids.reserve(internal_results[i].size());
        result.distances.reserve(internal_results[i].size());
        
        for (auto internal_idx : internal_results[i]) {
            auto it = impl_->reverse_id_map_.find(internal_idx);
            if (it != impl_->reverse_id_map_.end()) {
                result.ids.push_back(it->second);
                
                if (config.return_distances) {
                    song_kernel::dist_t dist = 0.0;
                    
                    switch (distance_metric_to_kernel_type(metric_)) {
                        case 0:
                            dist = impl_->data_->l2_distance(internal_idx, sparse_queries[i]);
                            break;
                        case 1:
                            dist = impl_->data_->negative_inner_prod_distance(internal_idx, sparse_queries[i]);
                            dist = -dist;
                            break;
                        case 2:
                            dist = impl_->data_->negative_cosine_distance(internal_idx, sparse_queries[i]);
                            dist = -dist;
                            break;
                    }
                    
                    result.distances.push_back(static_cast<float>(dist));
                }
            }
        }
        
        result.actual_k = result.ids.size();
        results.push_back(std::move(result));
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    metrics_.search_time_seconds += duration.count() / 1000000.0;
    
    for (const auto& res : results) {
        metrics_.distance_computations += res.actual_k;
    }
    
    return results;
}

void SongANNS::add_vector(const VectorEntry& entry) {
    if (!is_built_ || !impl_->graph_ || !impl_->data_) {
        throw std::runtime_error("SONG: cannot add vector before index is built");
    }
    
    if (entry.second.size() != static_cast<size_t>(dimension_)) {
        throw std::runtime_error("SONG: vector dimension mismatch");
    }
    
    if (impl_->next_internal_idx_ >= max_vectors_) {
        throw std::runtime_error("SONG: index capacity exceeded");
    }
    
    VectorId external_id = entry.first;
    if (impl_->id_map_.count(external_id)) {
        throw std::runtime_error("SONG: vector ID already exists");
    }
    
    song_kernel::idx_t internal_idx = impl_->next_internal_idx_++;
    impl_->id_map_[external_id] = internal_idx;
    impl_->reverse_id_map_[internal_idx] = external_id;
    
    auto sparse_vec = to_sparse_vector(entry.second);
    impl_->data_->add(internal_idx, sparse_vec);
    impl_->graph_->add_vertex(internal_idx, sparse_vec);
}

void SongANNS::add_vectors(const std::vector<VectorEntry>& entries) {
    for (const auto& entry : entries) {
        add_vector(entry);
    }
}

size_t SongANNS::get_index_size() const {
    return impl_->id_map_.size();
}

size_t SongANNS::get_memory_usage() const {
    size_t usage = 0;
    
    if (impl_->data_) {
        usage += max_vectors_ * static_cast<size_t>(dimension_) * sizeof(song_kernel::value_t);
    }
    
    if (impl_->graph_) {
        usage += max_vectors_ * 32 * sizeof(song_kernel::idx_t);
        usage += max_vectors_ * 32 * sizeof(song_kernel::dist_t);
    }
    
    usage += impl_->id_map_.size() * (sizeof(VectorId) + sizeof(song_kernel::idx_t));
    usage += impl_->reverse_id_map_.size() * (sizeof(song_kernel::idx_t) + sizeof(VectorId));
    
    return usage;
}

std::unordered_map<std::string, std::string> SongANNS::get_build_params() const {
    std::unordered_map<std::string, std::string> params;
    params["metric"] = std::to_string(static_cast<int>(metric_));
    params["dimension"] = std::to_string(dimension_);
    params["capacity"] = std::to_string(max_vectors_);
    return params;
}

ANNSMetrics SongANNS::get_metrics() const {
    return metrics_;
}

bool SongANNS::validate_params(const AlgorithmParams& params) const {
    (void)params;
    return true;
}

AlgorithmParams SongANNS::get_default_params() const {
    AlgorithmParams params;
    params.set("metric", static_cast<int>(DistanceMetric::L2));
    params.set("capacity", 100000);
    return params;
}

QueryConfig SongANNS::get_default_query_config() const {
    QueryConfig config;
    config.k = 10;
    config.return_distances = true;
    return config;
}

std::unique_ptr<ANNSAlgorithm> SongANNSFactory::create() const {
    return std::make_unique<SongANNS>();
}

std::string SongANNSFactory::algorithm_description() const {
    return "SONG GPU ANN backend";
}

std::vector<DistanceMetric> SongANNSFactory::supported_distances() const {
    return {DistanceMetric::L2, DistanceMetric::INNER_PRODUCT, DistanceMetric::COSINE};
}

AlgorithmParams SongANNSFactory::default_build_params() const {
    SongANNS instance;
    return instance.get_default_params();
}

QueryConfig SongANNSFactory::default_query_config() const {
    return SongANNS().get_default_query_config();
}

} // namespace anns
} // namespace sage_vdb

#endif // ENABLE_SONG

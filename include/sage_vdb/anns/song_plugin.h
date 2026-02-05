#pragma once

#include "sage_vdb/anns/anns_interface.h"

#ifdef ENABLE_SONG

namespace sage_vdb {
namespace anns {

/**
 * @brief SONG (GPU-accelerated graph-based ANN) implementation
 * 
 * Ports the legacy CUDA-based SONG algorithm without LibTorch dependencies.
 * Uses warp-level A* search on GPU for fast approximate nearest neighbor queries.
 * 
 * Requirements:
 * - CUDA runtime and cuBLAS
 * - Build with -DENABLE_SONG=ON
 */
class SongANNS : public ANNSAlgorithm {
public:
    SongANNS();
    ~SongANNS() override;
    
    // Algorithm identification
    std::string name() const override { return "SONG"; }
    std::string version() const override { return "1.0.0"; }
    std::string description() const override;
    
    // Capability queries
    std::vector<DistanceMetric> supported_distances() const override;
    bool supports_distance(DistanceMetric metric) const override;
    bool supports_updates() const override { return true; }
    bool supports_deletions() const override { return false; }
    bool supports_range_search() const override { return false; }
    
    // Index lifecycle
    void fit(const std::vector<VectorEntry>& dataset,
             const AlgorithmParams& params = {}) override;
    bool save(const std::string& path) const override;
    bool load(const std::string& path) override;
    bool is_built() const override;
    
    // Search operations
    ANNSResult query(const Vector& query_vector, 
                    const QueryConfig& config = {}) const override;
    
    std::vector<ANNSResult> batch_query(
        const std::vector<Vector>& query_vectors,
        const QueryConfig& config = {}) const override;
    
    // Update operations
    void add_vector(const VectorEntry& entry) override;
    void add_vectors(const std::vector<VectorEntry>& entries) override;
    
    // Statistics and introspection
    size_t get_index_size() const override;
    size_t get_memory_usage() const override;
    std::unordered_map<std::string, std::string> get_build_params() const override;
    ANNSMetrics get_metrics() const override;
    
    // Configuration
    bool validate_params(const AlgorithmParams& params) const override;
    AlgorithmParams get_default_params() const override;
    QueryConfig get_default_query_config() const override;

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
    
    DistanceMetric metric_;
    int dimension_;
    size_t max_vectors_;
    bool is_built_;
    mutable ANNSMetrics metrics_;
};

/**
 * @brief Factory for creating SONG ANNS instances
 */
class SongANNSFactory : public ANNSFactory {
public:
    std::unique_ptr<ANNSAlgorithm> create() const override;
    std::string algorithm_name() const override { return "SONG"; }
    std::string algorithm_description() const override;
    std::vector<DistanceMetric> supported_distances() const override;
    AlgorithmParams default_build_params() const override;
    QueryConfig default_query_config() const override;
};

} // namespace anns
} // namespace sage_vdb

#endif // ENABLE_SONG

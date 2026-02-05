#pragma once

#include "sage_vdb/anns/anns_interface.h"

namespace sage_vdb {
namespace anns {

/**
 * @brief CPU-based Vamana graph ANN algorithm.
 *
 * Ported from the legacy sage-db implementation and refactored to remove
 * LibTorch dependencies. Provides greedy graph search with robust pruning
 * and supports incremental insert/delete operations.
 */
class VamanaANNS : public ANNSAlgorithm {
public:
    VamanaANNS();
    ~VamanaANNS() override;

    // Identification
    std::string name() const override { return "Vamana"; }
    std::string version() const override;
    std::string description() const override;

    // Capabilities
    std::vector<DistanceMetric> supported_distances() const override;
    bool supports_distance(DistanceMetric metric) const override;
    bool supports_updates() const override { return true; }
    bool supports_deletions() const override { return true; }
    bool supports_range_search() const override { return false; }

    // Lifecycle
    void fit(const std::vector<VectorEntry>& dataset,
             const AlgorithmParams& params = {}) override;
    bool save(const std::string& path) const override;
    bool load(const std::string& path) override;
    bool is_built() const override { return built_; }

    // Query
    ANNSResult query(const Vector& query_vector,
                     const QueryConfig& config = {}) const override;
    std::vector<ANNSResult> batch_query(
        const std::vector<Vector>& query_vectors,
        const QueryConfig& config = {}) const override;

    // Mutations
    void add_vector(const VectorEntry& entry) override;
    void add_vectors(const std::vector<VectorEntry>& entries) override;
    void remove_vector(VectorId id) override;
    void remove_vectors(const std::vector<VectorId>& ids) override;

    // Stats
    size_t get_index_size() const override;
    size_t get_memory_usage() const override;
    std::unordered_map<std::string, std::string> get_build_params() const override;
    ANNSMetrics get_metrics() const override;

    // Configuration helpers
    bool validate_params(const AlgorithmParams& params) const override;
    AlgorithmParams get_default_params() const override;
    QueryConfig get_default_query_config() const override;

private:
    class Impl;
    std::unique_ptr<Impl> impl_;

    bool built_;
    AlgorithmParams build_params_;
    mutable ANNSMetrics metrics_;
};

class VamanaANNSFactory : public ANNSFactory {
public:
    std::unique_ptr<ANNSAlgorithm> create() const override;
    std::string algorithm_name() const override { return "Vamana"; }
    std::string algorithm_description() const override;
    std::vector<DistanceMetric> supported_distances() const override;
    AlgorithmParams default_build_params() const override;
    QueryConfig default_query_config() const override;
};

} // namespace anns
} // namespace sage_vdb

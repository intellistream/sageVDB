#pragma once

#include "sage_vdb/anns/anns_interface.h"
#include <unordered_map>

namespace sage_vdb {
namespace anns {

class BruteForceANNS : public ANNSAlgorithm {
public:
    BruteForceANNS();
    ~BruteForceANNS() override = default;

    std::string name() const override { return "brute_force"; }
    std::string version() const override;
    std::string description() const override;

    std::vector<DistanceMetric> supported_distances() const override;
    bool supports_distance(DistanceMetric metric) const override;
    bool supports_updates() const override { return true; }
    bool supports_deletions() const override { return true; }
    bool supports_range_search() const override { return true; }

    void fit(const std::vector<VectorEntry>& dataset,
             const AlgorithmParams& params = {}) override;
    bool save(const std::string& path) const override;
    bool load(const std::string& path) override;
    bool is_built() const override { return built_; }

    ANNSResult query(const Vector& query_vector,
                     const QueryConfig& config = {}) const override;
    std::vector<ANNSResult> batch_query(const std::vector<Vector>& query_vectors,
                                        const QueryConfig& config = {}) const override;
    ANNSResult range_query(const Vector& query_vector,
                           float radius,
                           const QueryConfig& config = {}) const override;

    void add_vector(const VectorEntry& entry) override;
    void add_vectors(const std::vector<VectorEntry>& entries) override;
    void remove_vector(VectorId id) override;
    void remove_vectors(const std::vector<VectorId>& ids) override;

    size_t get_index_size() const override;
    size_t get_memory_usage() const override;
    std::unordered_map<std::string, std::string> get_build_params() const override;
    ANNSMetrics get_metrics() const override { return metrics_; }

    bool validate_params(const AlgorithmParams& params) const override;
    AlgorithmParams get_default_params() const override;
    QueryConfig get_default_query_config() const override;

private:
    struct Entry {
        VectorId id;
        Vector vector;
    };

    float compute_distance(const Vector& a, const Vector& b) const;
    ANNSResult perform_query(const Vector& query_vector,
                             const QueryConfig& config) const;

    DistanceMetric metric_;
    Dimension dimension_;
    std::vector<Entry> dataset_;
    std::unordered_map<VectorId, size_t> id_to_index_;
    mutable ANNSMetrics metrics_;
    bool built_;
};

class BruteForceANNSFactory : public ANNSFactory {
public:
    std::unique_ptr<ANNSAlgorithm> create() const override;
    std::string algorithm_name() const override { return "brute_force"; }
    std::string algorithm_description() const override;
    std::vector<DistanceMetric> supported_distances() const override;
    AlgorithmParams default_build_params() const override;
    QueryConfig default_query_config() const override;
};

} // namespace anns
} // namespace sage_vdb
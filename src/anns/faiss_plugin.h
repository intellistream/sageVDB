#pragma once

#include "sage_vdb/anns/anns_interface.h"
#include <cstdint>

namespace faiss {
using idx_t = int64_t;
}

#include <faiss/MetricType.h>

#include <faiss/Index.h>
#include <faiss/impl/FaissAssert.h>

namespace sage_vdb {
namespace anns {

class FaissANNS : public ANNSAlgorithm {
public:
    FaissANNS();
    ~FaissANNS() override;

    std::string name() const override { return "FAISS"; }
    std::string version() const override;
    std::string description() const override;

    std::vector<DistanceMetric> supported_distances() const override;
    bool supports_distance(DistanceMetric metric) const override;
    bool supports_updates() const override { return true; }
    bool supports_deletions() const override { return false; }
    bool supports_range_search() const override { return true; }

    void fit(const std::vector<VectorEntry>& dataset,
             const AlgorithmParams& params = {}) override;
    bool save(const std::string& path) const override;
    bool load(const std::string& path) override;
    bool is_built() const override;

    ANNSResult query(const Vector& query_vector,
                     const QueryConfig& config = {}) const override;

    std::vector<ANNSResult> batch_query(
        const std::vector<Vector>& query_vectors,
        const QueryConfig& config = {}) const override;

    ANNSResult range_query(const Vector& query_vector,
                           float radius,
                           const QueryConfig& config = {}) const override;

    void add_vector(const VectorEntry& entry) override;
    void add_vectors(const std::vector<VectorEntry>& entries) override;

    size_t get_index_size() const override;
    size_t get_memory_usage() const override;
    std::unordered_map<std::string, std::string> get_build_params() const override;
    ANNSMetrics get_metrics() const override;

    bool validate_params(const AlgorithmParams& params) const override;
    AlgorithmParams get_default_params() const override;
    QueryConfig get_default_query_config() const override;

private:
    enum class IndexType {
        FLAT,
        IVF_FLAT,
        IVF_PQ,
        HNSW,
        AUTO
    };

    std::unique_ptr<faiss::Index> create_flat_index(int dim);
    std::unique_ptr<faiss::Index> create_ivf_flat_index(int dim, const AlgorithmParams& params);
    std::unique_ptr<faiss::Index> create_ivf_pq_index(int dim, const AlgorithmParams& params);
    std::unique_ptr<faiss::Index> create_hnsw_index(int dim, const AlgorithmParams& params);
    std::unique_ptr<faiss::Index> create_auto_index(int dim, size_t num_vectors,
                                                     const AlgorithmParams& params);

    IndexType parse_index_type(const std::string& type_str) const;
    DistanceMetric parse_distance_metric(const AlgorithmParams& params) const;
    std::string metric_to_string(DistanceMetric metric) const;
    std::string index_type_to_string(IndexType type) const;
    ANNSResult convert_faiss_results(const faiss::idx_t* ids,
                                     const float* distances,
                                     size_t k) const;
    void normalize_vector(std::vector<float>& vec) const;
    std::string metadata_path(const std::string& base_path) const;
    void apply_query_params(const QueryConfig& config) const;

    std::unique_ptr<faiss::Index> index_;

    DistanceMetric distance_metric_;
    int dimension_;
    IndexType index_type_;
    AlgorithmParams build_params_;
    size_t last_index_size_bytes_ = 0;

    mutable ANNSMetrics metrics_;
    bool is_built_;
};

class FaissANNSFactory : public ANNSFactory {
public:
    std::unique_ptr<ANNSAlgorithm> create() const override;
    std::string algorithm_name() const override { return "FAISS"; }
    std::string algorithm_description() const override;
    std::vector<DistanceMetric> supported_distances() const override;
    AlgorithmParams default_build_params() const override;
    QueryConfig default_query_config() const override;
};

} // namespace anns
} // namespace sage_vdb

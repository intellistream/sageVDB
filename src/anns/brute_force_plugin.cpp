#include "sage_vdb/anns/brute_force_plugin.h"
#include <fstream>
#include <algorithm>
#include <cmath>
#include <chrono>

namespace sage_vdb {
namespace anns {

namespace {
REGISTER_ANNS_ALGORITHM(BruteForceANNSFactory);
}

BruteForceANNS::BruteForceANNS()
    : metric_(DistanceMetric::L2), dimension_(0), built_(false) {
    metrics_.reset();
}

std::string BruteForceANNS::version() const {
    return "1.0.0";
}

std::string BruteForceANNS::description() const {
    return "Reference brute-force ANNS implementation for exact search";
}

std::vector<DistanceMetric> BruteForceANNS::supported_distances() const {
    return {DistanceMetric::L2, DistanceMetric::INNER_PRODUCT, DistanceMetric::COSINE};
}

bool BruteForceANNS::supports_distance(DistanceMetric metric) const {
    auto metrics = supported_distances();
    return std::find(metrics.begin(), metrics.end(), metric) != metrics.end();
}

void BruteForceANNS::fit(const std::vector<VectorEntry>& dataset,
                         const AlgorithmParams& params) {
    metrics_.reset();
    metric_ = static_cast<DistanceMetric>(
        params.get<int>("metric", static_cast<int>(DistanceMetric::L2))
    );

    dataset_.clear();
    id_to_index_.clear();

    if (dataset.empty()) {
        built_ = true;
        dimension_ = 0;
        return;
    }

    dimension_ = dataset.front().second.size();
    dataset_.reserve(dataset.size());

    auto start = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < dataset.size(); ++i) {
        const auto& entry = dataset[i];
        dataset_.push_back({entry.first, entry.second});
        id_to_index_[entry.first] = i;
    }
    auto end = std::chrono::high_resolution_clock::now();

    metrics_.build_time_seconds = std::chrono::duration<double>(end - start).count();
    metrics_.index_size_bytes = dataset_.size() * dimension_ * sizeof(float);
    built_ = true;
}

bool BruteForceANNS::save(const std::string& path) const {
    std::ofstream out(path, std::ios::binary);
    if (!out.is_open()) {
        return false;
    }

    out.write(reinterpret_cast<const char*>(&dimension_), sizeof(dimension_));
    int metric = static_cast<int>(metric_);
    out.write(reinterpret_cast<const char*>(&metric), sizeof(metric));

    uint64_t count = dataset_.size();
    out.write(reinterpret_cast<const char*>(&count), sizeof(count));
    for (const auto& entry : dataset_) {
        out.write(reinterpret_cast<const char*>(&entry.id), sizeof(entry.id));
        uint32_t dim = static_cast<uint32_t>(entry.vector.size());
        out.write(reinterpret_cast<const char*>(&dim), sizeof(dim));
        out.write(reinterpret_cast<const char*>(entry.vector.data()), dim * sizeof(float));
    }

    return true;
}

bool BruteForceANNS::load(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in.is_open()) {
        return false;
    }

    in.read(reinterpret_cast<char*>(&dimension_), sizeof(dimension_));
    int metric;
    in.read(reinterpret_cast<char*>(&metric), sizeof(metric));
    metric_ = static_cast<DistanceMetric>(metric);

    uint64_t count = 0;
    in.read(reinterpret_cast<char*>(&count), sizeof(count));

    dataset_.clear();
    id_to_index_.clear();
    dataset_.reserve(count);

    for (uint64_t i = 0; i < count; ++i) {
        Entry entry{};
        in.read(reinterpret_cast<char*>(&entry.id), sizeof(entry.id));
        uint32_t dim = 0;
        in.read(reinterpret_cast<char*>(&dim), sizeof(dim));
        entry.vector.resize(dim);
        in.read(reinterpret_cast<char*>(entry.vector.data()), dim * sizeof(float));
        dataset_.push_back(entry);
        id_to_index_[entry.id] = dataset_.size() - 1;
    }

    built_ = true;
    metrics_.reset();
    return true;
}

ANNSResult BruteForceANNS::query(const Vector& query_vector,
                                 const QueryConfig& config) const {
    return perform_query(query_vector, config);
}

std::vector<ANNSResult> BruteForceANNS::batch_query(
    const std::vector<Vector>& query_vectors,
    const QueryConfig& config) const {
    std::vector<ANNSResult> results;
    results.reserve(query_vectors.size());

    for (const auto& query : query_vectors) {
        results.push_back(perform_query(query, config));
    }

    return results;
}

ANNSResult BruteForceANNS::range_query(const Vector& query_vector,
                                       float radius,
                                       const QueryConfig& config) const {
    if (!built_) {
        throw std::runtime_error("BruteForceANNS index is not built");
    }

    ANNSResult result;
    result.actual_k = 0;

    auto start = std::chrono::high_resolution_clock::now();

    for (const auto& entry : dataset_) {
        float distance = compute_distance(query_vector, entry.vector);
        metrics_.distance_computations++;
        if (distance <= radius) {
            result.ids.push_back(entry.id);
            if (config.return_distances) {
                result.distances.push_back(distance);
            }
            result.actual_k++;
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    metrics_.search_time_seconds = std::chrono::duration<double>(end - start).count();

    return result;
}

void BruteForceANNS::add_vector(const VectorEntry& entry) {
    dataset_.push_back({entry.first, entry.second});
    id_to_index_[entry.first] = dataset_.size() - 1;
    built_ = true;
}

void BruteForceANNS::add_vectors(const std::vector<VectorEntry>& entries) {
    for (const auto& entry : entries) {
        add_vector(entry);
    }
}

void BruteForceANNS::remove_vector(VectorId id) {
    auto it = id_to_index_.find(id);
    if (it == id_to_index_.end()) {
        return;
    }

    size_t index = it->second;
    size_t last_index = dataset_.size() - 1;
    if (index != last_index) {
        std::swap(dataset_[index], dataset_[last_index]);
        id_to_index_[dataset_[index].id] = index;
    }
    dataset_.pop_back();
    id_to_index_.erase(it);
}

void BruteForceANNS::remove_vectors(const std::vector<VectorId>& ids) {
    for (auto id : ids) {
        remove_vector(id);
    }
}

size_t BruteForceANNS::get_index_size() const {
    return dataset_.size();
}

size_t BruteForceANNS::get_memory_usage() const {
    return dataset_.size() * dimension_ * sizeof(float) +
           id_to_index_.size() * (sizeof(VectorId) + sizeof(size_t));
}

std::unordered_map<std::string, std::string> BruteForceANNS::get_build_params() const {
    return {};
}

bool BruteForceANNS::validate_params(const AlgorithmParams& params) const {
    (void)params;
    return true;
}

AlgorithmParams BruteForceANNS::get_default_params() const {
    AlgorithmParams params;
    params.set("metric", static_cast<int>(DistanceMetric::L2));
    return params;
}

QueryConfig BruteForceANNS::get_default_query_config() const {
    QueryConfig config;
    config.k = 10;
    config.return_distances = true;
    return config;
}

float BruteForceANNS::compute_distance(const Vector& a, const Vector& b) const {
    if (a.size() != b.size()) {
        throw std::runtime_error("Vector dimension mismatch in BruteForceANNS");
    }

    switch (metric_) {
        case DistanceMetric::L2: {
            float distance = 0.0f;
            for (size_t i = 0; i < a.size(); ++i) {
                float diff = a[i] - b[i];
                distance += diff * diff;
            }
            return std::sqrt(distance);
        }
        case DistanceMetric::INNER_PRODUCT: {
            float dot = 0.0f;
            for (size_t i = 0; i < a.size(); ++i) {
                dot += a[i] * b[i];
            }
            return dot; // higher is better
        }
        case DistanceMetric::COSINE: {
            float dot = 0.0f;
            float norm_a = 0.0f;
            float norm_b = 0.0f;
            for (size_t i = 0; i < a.size(); ++i) {
                dot += a[i] * b[i];
                norm_a += a[i] * a[i];
                norm_b += b[i] * b[i];
            }
            float denom = std::sqrt(norm_a) * std::sqrt(norm_b);
            if (denom == 0.0f) {
                return 1.0f;
            }
            return 1.0f - (dot / denom);
        }
    }
    return 0.0f;
}

ANNSResult BruteForceANNS::perform_query(const Vector& query_vector,
                                         const QueryConfig& config) const {
    if (!built_) {
        throw std::runtime_error("BruteForceANNS index is not built");
    }

    std::vector<std::pair<float, VectorId>> scored_results;
    scored_results.reserve(dataset_.size());

    auto start = std::chrono::high_resolution_clock::now();

    for (const auto& entry : dataset_) {
        float distance = compute_distance(query_vector, entry.vector);
        metrics_.distance_computations++;
        scored_results.emplace_back(distance, entry.id);
    }

    auto comparator = [this](const auto& a, const auto& b) {
        if (metric_ == DistanceMetric::INNER_PRODUCT) {
            return a.first > b.first; // higher is better
        }
        return a.first < b.first; // lower is better
    };

    uint32_t k = std::min(static_cast<uint32_t>(scored_results.size()), config.k);
    std::partial_sort(scored_results.begin(), scored_results.begin() + k,
                      scored_results.end(), comparator);

    auto end = std::chrono::high_resolution_clock::now();
    metrics_.search_time_seconds = std::chrono::duration<double>(end - start).count();

    ANNSResult result;
    result.ids.reserve(k);
    if (config.return_distances) {
        result.distances.reserve(k);
    }

    for (uint32_t i = 0; i < k; ++i) {
        result.ids.push_back(scored_results[i].second);
        if (config.return_distances) {
            result.distances.push_back(scored_results[i].first);
        }
    }
    result.actual_k = k;

    return result;
}

std::unique_ptr<ANNSAlgorithm> BruteForceANNSFactory::create() const {
    auto algorithm = std::make_unique<BruteForceANNS>();
    return algorithm;
}

std::string BruteForceANNSFactory::algorithm_description() const {
    return "Brute-force exact nearest neighbor search";
}

std::vector<DistanceMetric> BruteForceANNSFactory::supported_distances() const {
    return {DistanceMetric::L2, DistanceMetric::INNER_PRODUCT, DistanceMetric::COSINE};
}

AlgorithmParams BruteForceANNSFactory::default_build_params() const {
    AlgorithmParams params;
    params.set("metric", static_cast<int>(DistanceMetric::L2));
    return params;
}

QueryConfig BruteForceANNSFactory::default_query_config() const {
    QueryConfig config;
    config.k = 10;
    config.return_distances = true;
    return config;
}

} // namespace anns
} // namespace sage_vdb

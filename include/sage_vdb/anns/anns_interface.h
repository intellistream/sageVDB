#pragma once

#include "sage_vdb/common.h"
#include <memory>
#include <string>
#include <vector>
#include <unordered_map>
#include <algorithm>
#include <chrono>
#include <functional>
#include <utility>
#include <stdexcept>
#include <type_traits>

namespace sage_vdb {
namespace anns {

/**
 * @brief Algorithm build parameters following big-ann-benchmarks style
 */
struct AlgorithmParams {
    std::unordered_map<std::string, std::string> params;
    
    template<typename T>
    T get(const std::string& key, const T& default_value = T{}) const {
        auto it = params.find(key);
        if (it == params.end()) {
            return default_value;
        }
        return parse_value<T>(it->second);
    }
    
    template<typename T>
    void set(const std::string& key, const T& value) {
        params[key] = stringify_value(value);
    }
    
    void set_raw(const std::string& key, const std::string& value) {
        params[key] = value;
    }

    bool has(const std::string& key) const {
        return params.find(key) != params.end();
    }
    
private:
    template<typename T>
    T parse_value(const std::string& str) const;
    
    template<typename T>
    std::string stringify_value(const T& value) const;
};

/**
 * @brief Query configuration for search operations
 */
struct QueryConfig {
    uint32_t k = 10;                    // Number of nearest neighbors
    bool return_distances = true;       // Whether to return distances
    AlgorithmParams algorithm_params;   // Algorithm-specific parameters
    
    template<typename T>
    T get_param(const std::string& key, const T& default_value = T{}) const {
        return algorithm_params.get(key, default_value);
    }
    
    template<typename T>
    void set_param(const std::string& key, const T& value) {
        algorithm_params.set(key, value);
    }

    void set_raw_param(const std::string& key, const std::string& value) {
        algorithm_params.set_raw(key, value);
    }
};

/**
 * @brief Search result from ANNS algorithm
 */
struct ANNSResult {
    std::vector<VectorId> ids;
    std::vector<float> distances;
    size_t actual_k = 0;
    
    ANNSResult() = default;
    ANNSResult(std::vector<VectorId> ids_, std::vector<float> distances_)
        : ids(std::move(ids_)), distances(std::move(distances_)), actual_k(ids.size()) {}
};

/**
 * @brief Performance metrics for ANNS operations
 */
struct ANNSMetrics {
    double build_time_seconds = 0.0;
    double search_time_seconds = 0.0;
    size_t index_size_bytes = 0;
    size_t distance_computations = 0;
    std::unordered_map<std::string, double> additional_metrics;
    
    void reset() {
        build_time_seconds = 0.0;
        search_time_seconds = 0.0;
        index_size_bytes = 0;
        distance_computations = 0;
        additional_metrics.clear();
    }
    
    void merge(const ANNSMetrics& other) {
        build_time_seconds += other.build_time_seconds;
        search_time_seconds += other.search_time_seconds;
        index_size_bytes = std::max(index_size_bytes, other.index_size_bytes);
        distance_computations += other.distance_computations;
        for (const auto& [key, value] : other.additional_metrics) {
            additional_metrics[key] += value;
        }
    }
};

using VectorEntry = std::pair<VectorId, Vector>;

/**
 * @brief Base interface for all ANNS algorithms
 * 
 * Design inspired by big-ann-benchmarks for maximum compatibility
 * and ease of integration with external algorithms.
 */
class ANNSAlgorithm {
public:
    virtual ~ANNSAlgorithm() = default;
    
    // Algorithm identification
    virtual std::string name() const = 0;
    virtual std::string version() const = 0;
    virtual std::string description() const = 0;
    
    // Capability queries
    virtual std::vector<DistanceMetric> supported_distances() const = 0;
    virtual bool supports_distance(DistanceMetric metric) const = 0;
    virtual bool supports_updates() const = 0;
    virtual bool supports_deletions() const = 0;
    virtual bool supports_range_search() const = 0;
    
    // Index lifecycle
    virtual void fit(const std::vector<VectorEntry>& dataset, 
                    const AlgorithmParams& params = {}) = 0;
    virtual bool save(const std::string& path) const = 0;
    virtual bool load(const std::string& path) = 0;
    virtual bool is_built() const = 0;
    
    // Search operations
    virtual ANNSResult query(const Vector& query_vector, 
                           const QueryConfig& config = {}) const = 0;
    
    virtual std::vector<ANNSResult> batch_query(
        const std::vector<Vector>& query_vectors,
        const QueryConfig& config = {}) const = 0;
    
    // Optional operations (throw std::runtime_error if not supported)
    virtual ANNSResult range_query(const Vector& query_vector, 
                                  float radius,
                                  const QueryConfig& config = {}) const {
        (void)query_vector;
        (void)radius;
        (void)config;
        throw std::runtime_error(name() + " does not support range queries");
    }
    
    virtual void add_vector(const VectorEntry& entry) {
        (void)entry;
        throw std::runtime_error(name() + " does not support adding vectors");
    }
    
    virtual void add_vectors(const std::vector<VectorEntry>& entries) {
        (void)entries;
        throw std::runtime_error(name() + " does not support adding vectors");
    }
    
    virtual void remove_vector(VectorId id) {
        (void)id;
        throw std::runtime_error(name() + " does not support removing vectors");
    }
    
    virtual void remove_vectors(const std::vector<VectorId>& ids) {
        (void)ids;
        throw std::runtime_error(name() + " does not support removing vectors");
    }
    
    // Statistics and introspection
    virtual size_t get_index_size() const = 0;
    virtual size_t get_memory_usage() const = 0;
    virtual std::unordered_map<std::string, std::string> get_build_params() const = 0;
    virtual ANNSMetrics get_metrics() const = 0;
    
    // Configuration validation
    virtual bool validate_params(const AlgorithmParams& params) const = 0;
    virtual AlgorithmParams get_default_params() const = 0;
    virtual QueryConfig get_default_query_config() const = 0;
    
protected:
    // Utility for timing operations
    template<typename Func>
    std::pair<typename std::invoke_result<Func>::type, double> 
    time_operation(Func&& func) const {
        auto start = std::chrono::high_resolution_clock::now();
        auto result = func();
        auto end = std::chrono::high_resolution_clock::now();
        
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        double seconds = duration.count() / 1000000.0;
        
        return std::make_pair(std::move(result), seconds);
    }
};

/**
 * @brief Factory interface for creating ANNS algorithm instances
 */
class ANNSFactory {
public:
    virtual ~ANNSFactory() = default;
    
    virtual std::unique_ptr<ANNSAlgorithm> create() const = 0;
    virtual std::string algorithm_name() const = 0;
    virtual std::string algorithm_description() const = 0;
    virtual std::vector<DistanceMetric> supported_distances() const = 0;
    virtual AlgorithmParams default_build_params() const = 0;
    virtual QueryConfig default_query_config() const = 0;
};

/**
 * @brief Registry for ANNS algorithm factories
 * 
 * Singleton registry that manages available ANNS algorithms
 * Supports dynamic registration for plugin-style architecture
 */
class ANNSRegistry {
public:
    static ANNSRegistry& instance();
    
    // Factory management
    void register_factory(std::unique_ptr<ANNSFactory> factory);
    void register_factory(const std::string& name, std::unique_ptr<ANNSFactory> factory);
    void unregister_factory(const std::string& name);
    
    // Algorithm creation
    std::unique_ptr<ANNSAlgorithm> create_algorithm(const std::string& name) const;
    
    // Query capabilities
    std::vector<std::string> list_algorithms() const;
    bool is_available(const std::string& name) const;
    const ANNSFactory* get_factory(const std::string& name) const;
    
    // Capability queries
    std::vector<std::string> algorithms_supporting_distance(DistanceMetric metric) const;
    std::vector<std::string> algorithms_supporting_updates() const;
    std::vector<std::string> algorithms_supporting_deletions() const;
    
private:
    ANNSRegistry() = default;
    std::unordered_map<std::string, std::unique_ptr<ANNSFactory>> factories_;
};

/**
 * @brief Automatic registration helper for ANNS algorithms
 * 
 * Use this macro to automatically register an algorithm factory:
 * REGISTER_ANNS_ALGORITHM(MyAlgorithmFactory);
 */
class ANNSAutoRegistrar {
public:
    template<typename FactoryType>
    ANNSAutoRegistrar(FactoryType* factory) {
        ANNSRegistry::instance().register_factory(
            std::unique_ptr<ANNSFactory>(factory)
        );
    }
};

#define REGISTER_ANNS_ALGORITHM(factory_class) \
    namespace { \
        static ANNSAutoRegistrar _anns_registrar_##factory_class( \
            new factory_class() \
        ); \
    }

// Template implementations
template<typename T>
T AlgorithmParams::parse_value(const std::string& str) const {
    if constexpr (std::is_same_v<T, int>) {
        return std::stoi(str);
    } else if constexpr (std::is_same_v<T, unsigned int>) {
        return static_cast<unsigned int>(std::stoul(str));
    } else if constexpr (std::is_same_v<T, long>) {
        return std::stol(str);
    } else if constexpr (std::is_same_v<T, unsigned long>) {
        return std::stoul(str);
    } else if constexpr (std::is_same_v<T, float>) {
        return std::stof(str);
    } else if constexpr (std::is_same_v<T, double>) {
        return std::stod(str);
    } else if constexpr (std::is_same_v<T, bool>) {
        return str == "true" || str == "1" || str == "yes";
    } else if constexpr (std::is_same_v<T, std::string>) {
        return str;
    } else {
        static_assert(sizeof(T) == 0, "Unsupported parameter type");
    }
}

template<typename T>
std::string AlgorithmParams::stringify_value(const T& value) const {
    if constexpr (std::is_same_v<T, std::string>) {
        return value;
    } else if constexpr (std::is_same_v<T, bool>) {
        return value ? "true" : "false";
    } else {
        return std::to_string(value);
    }
}

} // namespace anns
} // namespace sage_vdb
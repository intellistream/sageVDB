#include "sage_vdb/vector_store.h"
#include "sage_vdb/anns/anns_interface.h"
#include "sage_vdb/anns/brute_force_plugin.h"
#ifdef ENABLE_FAISS
#include "sage_vdb/anns/faiss_plugin.h"
#endif

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <fstream>
#include <mutex>
#include <shared_mutex>
#include <stdexcept>
#include <unordered_map>

namespace sage_vdb {
namespace {
constexpr uint32_t kVectorStoreFormatVersion = 1;
}

class VectorStore::Impl {
public:
    explicit Impl(const DatabaseConfig& config)
        : config_(config), algorithm_name_(select_algorithm(config.anns_algorithm)) {
        initialize_algorithm();
    }

    VectorId add_vector(const Vector& vector) {
        VectorId id = next_id_++;
        dataset_.push_back({id, vector});
        id_to_index_[id] = dataset_.size() - 1;

        if (index_built_ && algorithm_->supports_updates()) {
            algorithm_->add_vector(dataset_.back());
        } else {
            index_dirty_ = true;
        }
        return id;
    }

    std::vector<VectorId> add_vectors(const std::vector<Vector>& vectors) {
        std::vector<VectorId> ids;
        ids.reserve(vectors.size());

        std::vector<anns::VectorEntry> entries;
        entries.reserve(vectors.size());

        for (const auto& vec : vectors) {
            VectorId id = next_id_++;
            ids.push_back(id);
            dataset_.push_back({id, vec});
            id_to_index_[id] = dataset_.size() - 1;
            entries.emplace_back(id, vec);
        }

        if (index_built_ && !entries.empty() && algorithm_->supports_updates()) {
            algorithm_->add_vectors(entries);
        } else if (!entries.empty()) {
            index_dirty_ = true;
        }

        return ids;
    }

    bool remove_vector(VectorId id) {
        auto it = id_to_index_.find(id);
        if (it == id_to_index_.end()) {
            return false;
        }

        size_t index = it->second;
        size_t last_index = dataset_.size() - 1;

        if (index != last_index) {
            std::swap(dataset_[index], dataset_[last_index]);
            id_to_index_[dataset_[index].first] = index;
        }

        dataset_.pop_back();
        id_to_index_.erase(it);

        if (index_built_ && algorithm_->supports_deletions()) {
            algorithm_->remove_vector(id);
        } else {
            index_dirty_ = true;
        }
        return true;
    }

    bool update_vector(VectorId id, const Vector& vector) {
        auto it = id_to_index_.find(id);
        if (it == id_to_index_.end()) {
            return false;
        }

        dataset_[it->second].second = vector;

        if (index_built_ && algorithm_->supports_updates() && algorithm_->supports_deletions()) {
            algorithm_->remove_vector(id);
            algorithm_->add_vector({id, vector});
        } else {
            index_dirty_ = true;
        }
        return true;
    }

    std::vector<QueryResult> search(const Vector& query, const SearchParams& params) {
        if (dataset_.empty()) {
            return {};
        }

        ensure_index_ready();

        auto query_config = create_query_config(params);

        if (params.radius > 0.0f && algorithm_->supports_range_search()) {
            auto range_result = execute_range_query(query, params.radius, query_config);
            last_metrics_ = algorithm_->get_metrics();
            return convert_result(range_result);
        }

        auto result = execute_query(query, query_config);
        last_metrics_ = algorithm_->get_metrics();
        return convert_result(result);
    }

    std::vector<std::vector<QueryResult>> batch_search(const std::vector<Vector>& queries,
                                                       const SearchParams& params) {
        if (queries.empty()) {
            return {};
        }
        ensure_index_ready();
        auto query_config = create_query_config(params);

    auto batch_results = execute_batch_query(queries, query_config);
    last_metrics_ = algorithm_->get_metrics();

        std::vector<std::vector<QueryResult>> converted;
        converted.reserve(batch_results.size());
        for (const auto& res : batch_results) {
            converted.emplace_back(convert_result(res));
        }
        return converted;
    }

    void build_index() {
        if (dataset_.empty()) {
            index_built_ = true;
            index_dirty_ = false;
            return;
        }

        auto build_params = compose_build_params();
        algorithm_->fit(dataset_, build_params);
        index_built_ = algorithm_->is_built();
        index_dirty_ = false;
    }

    void set_training_data(const std::vector<Vector>& training) {
        training_data_ = training;
        index_dirty_ = true;
    }

    bool is_trained() const {
        return index_built_ && !index_dirty_ && algorithm_->is_built();
    }

    size_t size() const {
        return dataset_.size();
    }

    void save(const std::string& filepath) const {
        std::ofstream out(filepath, std::ios::binary);
        if (!out.is_open()) {
            throw SageVDBException("Failed to open file for saving vector store: " + filepath);
        }

        uint32_t version = kVectorStoreFormatVersion;
        out.write(reinterpret_cast<const char*>(&version), sizeof(version));

        uint32_t name_length = static_cast<uint32_t>(algorithm_name_.size());
        out.write(reinterpret_cast<const char*>(&name_length), sizeof(name_length));
        out.write(algorithm_name_.data(), name_length);

        int metric_value = static_cast<int>(config_.metric);
        out.write(reinterpret_cast<const char*>(&metric_value), sizeof(metric_value));
        out.write(reinterpret_cast<const char*>(&config_.dimension), sizeof(config_.dimension));

        uint64_t count = dataset_.size();
        out.write(reinterpret_cast<const char*>(&count), sizeof(count));
        for (const auto& entry : dataset_) {
            out.write(reinterpret_cast<const char*>(&entry.first), sizeof(entry.first));
            Dimension dim = static_cast<Dimension>(entry.second.size());
            out.write(reinterpret_cast<const char*>(&dim), sizeof(dim));
            out.write(reinterpret_cast<const char*>(entry.second.data()), dim * sizeof(float));
        }

        out.write(reinterpret_cast<const char*>(&next_id_), sizeof(next_id_));
        out.close();

        std::string index_path = filepath + ".anns";
        if (algorithm_ && index_built_ && !index_dirty_) {
            algorithm_->save(index_path);
        } else {
            std::remove(index_path.c_str());
        }
    }

    void load(const std::string& filepath) {
        std::ifstream in(filepath, std::ios::binary);
        if (!in.is_open()) {
            throw SageVDBException("Failed to open file for loading vector store: " + filepath);
        }

        uint32_t version = 0;
        in.read(reinterpret_cast<char*>(&version), sizeof(version));
        if (version != kVectorStoreFormatVersion) {
            throw SageVDBException("Unsupported vector store format version");
        }

        uint32_t name_length = 0;
        in.read(reinterpret_cast<char*>(&name_length), sizeof(name_length));
        std::string stored_name(name_length, '\0');
        in.read(stored_name.data(), name_length);

        if (stored_name != algorithm_name_) {
            algorithm_name_ = select_algorithm(stored_name);
            initialize_algorithm();
        }

        int metric_value = 0;
        in.read(reinterpret_cast<char*>(&metric_value), sizeof(metric_value));
        config_.metric = static_cast<DistanceMetric>(metric_value);
        in.read(reinterpret_cast<char*>(&config_.dimension), sizeof(config_.dimension));

        uint64_t count = 0;
        in.read(reinterpret_cast<char*>(&count), sizeof(count));

        dataset_.clear();
        id_to_index_.clear();
        dataset_.reserve(count);

        for (uint64_t i = 0; i < count; ++i) {
            VectorId id;
            Dimension dim;
            in.read(reinterpret_cast<char*>(&id), sizeof(id));
            in.read(reinterpret_cast<char*>(&dim), sizeof(dim));
            Vector vec(dim);
            in.read(reinterpret_cast<char*>(vec.data()), dim * sizeof(float));
            dataset_.emplace_back(id, std::move(vec));
            id_to_index_[id] = dataset_.size() - 1;
        }

        in.read(reinterpret_cast<char*>(&next_id_), sizeof(next_id_));
        if (next_id_ <= 1 && !dataset_.empty()) {
            next_id_ = dataset_.back().first + 1;
        }

        in.close();

        std::string index_path = filepath + ".anns";
        bool loaded_index = algorithm_->load(index_path);
        index_built_ = loaded_index;
        index_dirty_ = !loaded_index;

        if (loaded_index) {
            size_t algo_size = algorithm_->get_index_size();
            if (algo_size != dataset_.size()) {
                index_built_ = false;
                index_dirty_ = true;
            }
        }
    }

    const DatabaseConfig& config() const { return config_; }

    void ensure_index_ready() {
        if (!index_built_ || index_dirty_) {
            build_index();
        }
    }

private:
    static std::string select_algorithm(const std::string& requested) {
        if (requested.empty() || requested == "AUTO" || requested == "auto") {
            return "brute_force";
        }
        return requested;
    }

    void initialize_algorithm() {
        auto& registry = anns::ANNSRegistry::instance();
        if (!registry.is_available(algorithm_name_)) {
            algorithm_name_ = "brute_force";
        }

        const auto* factory = registry.get_factory(algorithm_name_);
        if (!factory) {
            throw SageVDBException("Failed to locate ANNS factory for algorithm: " + algorithm_name_);
        }

        base_build_params_ = factory->default_build_params();
        base_query_config_ = factory->default_query_config();
        algorithm_ = factory->create();
        index_built_ = false;
        index_dirty_ = true;
    }

    anns::AlgorithmParams compose_build_params() const {
        auto params = base_build_params_;
        params.set("metric", static_cast<int>(config_.metric));
        params.set("dimension", static_cast<int>(config_.dimension));
        for (const auto& kv : config_.anns_build_params) {
            params.set_raw(kv.first, kv.second);
        }
        if (!training_data_.empty()) {
            params.set("training_size", static_cast<int>(training_data_.size()));
        }
        return params;
    }

    anns::QueryConfig create_query_config(const SearchParams& search_params) const {
        auto query_config = base_query_config_;
        query_config.k = search_params.k;
        query_config.return_distances = true;
        for (const auto& kv : config_.anns_query_params) {
            query_config.set_raw_param(kv.first, kv.second);
        }
        if (search_params.nprobe > 0) {
            query_config.set_param("nprobe", static_cast<int>(search_params.nprobe));
        }
        if (search_params.radius > 0.0f) {
            query_config.set_param("radius", search_params.radius);
        }
        return query_config;
    }

    anns::ANNSResult execute_query(
        const Vector& query, const anns::QueryConfig& config) const {
        return algorithm_->query(query, config);
    }

    anns::ANNSResult execute_range_query(
        const Vector& query, float radius, const anns::QueryConfig& config) const {
        return algorithm_->range_query(query, radius, config);
    }

    std::vector<anns::ANNSResult> execute_batch_query(
        const std::vector<Vector>& queries, const anns::QueryConfig& config) const {
        return algorithm_->batch_query(queries, config);
    }

    std::vector<QueryResult> convert_result(const anns::ANNSResult& result) const {
        std::vector<QueryResult> converted;
        converted.reserve(result.ids.size());
        for (size_t i = 0; i < result.ids.size(); ++i) {
            float score = (i < result.distances.size()) ? result.distances[i] : 0.0f;
            converted.emplace_back(result.ids[i], score);
        }
        return converted;
    }

    DatabaseConfig config_;
    std::string algorithm_name_;
    std::unique_ptr<anns::ANNSAlgorithm> algorithm_;
    anns::AlgorithmParams base_build_params_;
    anns::QueryConfig base_query_config_;
    std::vector<anns::VectorEntry> dataset_;
    std::unordered_map<VectorId, size_t> id_to_index_;
    std::vector<Vector> training_data_;
    anns::ANNSMetrics last_metrics_;
    bool index_built_ = false;
    bool index_dirty_ = true;
    VectorId next_id_ = 1;
};

VectorStore::VectorStore(const DatabaseConfig& config)
    : impl_(std::make_unique<Impl>(config)), config_(config) {}

VectorStore::~VectorStore() = default;

VectorId VectorStore::add_vector(const Vector& vector) {
    std::unique_lock<std::shared_mutex> lock(mutex_);  // Exclusive write lock
    validate_vector(vector);
    return impl_->add_vector(vector);
}

bool VectorStore::remove_vector(VectorId id) {
    std::unique_lock<std::shared_mutex> lock(mutex_);  // Exclusive write lock
    return impl_->remove_vector(id);
}

bool VectorStore::update_vector(VectorId id, const Vector& vector) {
    std::unique_lock<std::shared_mutex> lock(mutex_);  // Exclusive write lock
    validate_vector(vector);
    return impl_->update_vector(id, vector);
}

std::vector<VectorId> VectorStore::add_vectors(const std::vector<Vector>& vectors) {
    std::unique_lock<std::shared_mutex> lock(mutex_);  // Exclusive write lock
    for (const auto& vec : vectors) {
        validate_vector(vec);
    }
    return impl_->add_vectors(vectors);
}

std::vector<QueryResult> VectorStore::search(const Vector& query, const SearchParams& params) const {
    std::shared_lock<std::shared_mutex> lock(mutex_);  // Allow concurrent reads!
    validate_vector(query);
    return impl_->search(query, params);
}

std::vector<std::vector<QueryResult>> VectorStore::batch_search(
    const std::vector<Vector>& queries, const SearchParams& params) const {
    std::shared_lock<std::shared_mutex> lock(mutex_);  // Allow concurrent reads!
    for (const auto& query : queries) {
        validate_vector(query);
    }
    return impl_->batch_search(queries, params);
}

void VectorStore::build_index() {
    std::unique_lock<std::shared_mutex> lock(mutex_);  // Exclusive write lock
    impl_->build_index();
}

void VectorStore::train_index(const std::vector<Vector>& training_data) {
    std::unique_lock<std::shared_mutex> lock(mutex_);  // Exclusive write lock
    impl_->set_training_data(training_data);
}

bool VectorStore::is_trained() const {
    std::shared_lock<std::shared_mutex> lock(mutex_);  // Allow concurrent reads!
    return impl_->is_trained();
}

size_t VectorStore::size() const {
    std::shared_lock<std::shared_mutex> lock(mutex_);  // Allow concurrent reads!
    return impl_->size();
}

Dimension VectorStore::dimension() const {
    return config_.dimension;
}

IndexType VectorStore::index_type() const {
    return config_.index_type;
}

void VectorStore::save(const std::string& filepath) const {
    std::shared_lock<std::shared_mutex> lock(mutex_);  // Read-only operation
    impl_->save(filepath);
}

void VectorStore::load(const std::string& filepath) {
    std::unique_lock<std::shared_mutex> lock(mutex_);  // Exclusive write lock
    impl_->load(filepath);
    config_ = impl_->config();
}

void VectorStore::validate_vector(const Vector& vector) const {
    if (config_.dimension == 0) {
        throw SageVDBException("Database dimension is not configured");
    }
    if (vector.size() != config_.dimension) {
        throw SageVDBException("Vector dimension mismatch: expected " +
                             std::to_string(config_.dimension) +
                             ", got " + std::to_string(vector.size()));
    }
}

void VectorStore::ensure_trained() const {
    if (!is_trained()) {
        throw SageVDBException("Index is not trained. Call build_index() first.");
    }
}

} // namespace sage_vdb

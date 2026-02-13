#include "sage_vdb/sage_vdb.h"
#include <algorithm>
#include <fstream>

namespace sage_vdb {

SageVDB::SageVDB(const DatabaseConfig& config) : config_(config) {
    if (config_.dimension == 0) {
        throw SageVDBException("Vector dimension must be greater than 0");
    }
    
    // Create components
    vector_store_ = std::make_shared<VectorStore>(config_);
    metadata_store_ = std::make_shared<MetadataStore>();
    query_engine_ = std::make_shared<QueryEngine>(vector_store_, metadata_store_);
}

VectorId SageVDB::add(const Vector& vector, const Metadata& metadata) {
    validate_dimension(vector);
    
    VectorId id = vector_store_->add_vector(vector);
    
    if (!metadata.empty()) {
        metadata_store_->set_metadata(id, metadata);
    }
    
    return id;
}

std::vector<VectorId> SageVDB::add_batch(const std::vector<Vector>& vectors,
                                       const std::vector<Metadata>& metadata) {
    if (!metadata.empty()) {
        ensure_consistent_metadata(vectors, metadata);
    }
    
    for (const auto& vector : vectors) {
        validate_dimension(vector);
    }
    
    auto ids = vector_store_->add_vectors(vectors);
    
    if (!metadata.empty()) {
        metadata_store_->set_batch_metadata(ids, metadata);
    }
    
    return ids;
}

bool SageVDB::remove(VectorId id) {
    bool removed = vector_store_->remove_vector(id);
    metadata_store_->remove_metadata(id);
    return removed;
}

bool SageVDB::update(VectorId id, const Vector& vector, const Metadata& metadata) {
    bool updated = false;
    if (!vector.empty()) {
        validate_dimension(vector);
        updated = vector_store_->update_vector(id, vector);
    }
    
    if (!metadata.empty()) {
        metadata_store_->set_metadata(id, metadata);
        updated = true;
    }
    
    return updated;
}

std::vector<QueryResult> SageVDB::search(const Vector& query, 
                                       uint32_t k, 
                                       bool include_metadata) const {
    SearchParams params;
    params.k = k;
    params.include_metadata = include_metadata;
    return search(query, params);
}

std::vector<QueryResult> SageVDB::search(const Vector& query, const SearchParams& params) const {
    validate_dimension(query);
    return query_engine_->search(query, params);
}

std::vector<QueryResult> SageVDB::filtered_search(
    const Vector& query,
    const SearchParams& params,
    const std::function<bool(const Metadata&)>& filter) const {
    
    validate_dimension(query);
    return query_engine_->filtered_search(query, params, filter);
}

std::vector<std::vector<QueryResult>> SageVDB::batch_search(
    const std::vector<Vector>& queries, const SearchParams& params) const {
    
    for (const auto& query : queries) {
        validate_dimension(query);
    }
    
    return query_engine_->batch_search(queries, params);
}

void SageVDB::build_index() {
    vector_store_->build_index();
}

void SageVDB::train_index(const std::vector<Vector>& training_data) {
    if (!training_data.empty()) {
        for (const auto& vector : training_data) {
            validate_dimension(vector);
        }
    }
    vector_store_->train_index(training_data);
}

bool SageVDB::is_trained() const {
    return vector_store_->is_trained();
}

bool SageVDB::set_metadata(VectorId id, const Metadata& metadata) {
    metadata_store_->set_metadata(id, metadata);
    return true;
}

bool SageVDB::get_metadata(VectorId id, Metadata& metadata) const {
    return metadata_store_->get_metadata(id, metadata);
}

std::vector<VectorId> SageVDB::find_by_metadata(const std::string& key, 
                                               const MetadataValue& value) const {
    return metadata_store_->find_by_metadata(key, value);
}

void SageVDB::save(const std::string& filepath) const {
    vector_store_->save(filepath + ".vectors");
    metadata_store_->save(filepath + ".metadata");
    
    // Save configuration
    std::ofstream config_file(filepath + ".config");
    if (config_file.is_open()) {
        config_file << "dimension=" << config_.dimension << "\n";
        config_file << "index_type=" << static_cast<int>(config_.index_type) << "\n";
        config_file << "metric=" << static_cast<int>(config_.metric) << "\n";
        config_file << "nlist=" << config_.nlist << "\n";
        config_file << "m=" << config_.m << "\n";
        config_file << "nbits=" << config_.nbits << "\n";
        config_file << "M=" << config_.M << "\n";
        config_file << "efConstruction=" << config_.efConstruction << "\n";
    }
}

void SageVDB::load(const std::string& filepath) {
    // Load configuration
    std::ifstream config_file(filepath + ".config");
    if (config_file.is_open()) {
        std::string line;
        while (std::getline(config_file, line)) {
            size_t pos = line.find('=');
            if (pos != std::string::npos) {
                std::string key = line.substr(0, pos);
                std::string value = line.substr(pos + 1);
                
                if (key == "dimension") {
                    config_.dimension = std::stoul(value);
                } else if (key == "index_type") {
                    config_.index_type = static_cast<IndexType>(std::stoi(value));
                } else if (key == "metric") {
                    config_.metric = static_cast<DistanceMetric>(std::stoi(value));
                } else if (key == "nlist") {
                    config_.nlist = std::stoul(value);
                } else if (key == "m") {
                    config_.m = std::stoul(value);
                } else if (key == "nbits") {
                    config_.nbits = std::stoul(value);
                } else if (key == "M") {
                    config_.M = std::stoul(value);
                } else if (key == "efConstruction") {
                    config_.efConstruction = std::stoul(value);
                }
            }
        }
    }
    
    // Recreate components with loaded configuration
    vector_store_ = std::make_shared<VectorStore>(config_);
    metadata_store_ = std::make_shared<MetadataStore>();
    query_engine_ = std::make_shared<QueryEngine>(vector_store_, metadata_store_);
    
    // Load data
    vector_store_->load(filepath + ".vectors");
    metadata_store_->load(filepath + ".metadata");
}

size_t SageVDB::size() const {
    return vector_store_->size();
}

Dimension SageVDB::dimension() const {
    return config_.dimension;
}

IndexType SageVDB::index_type() const {
    return config_.index_type;
}

const DatabaseConfig& SageVDB::config() const {
    return config_;
}

void SageVDB::validate_dimension(const Vector& vector) const {
    if (vector.size() != config_.dimension) {
        throw SageVDBException("Vector dimension mismatch: expected " + 
                             std::to_string(config_.dimension) + 
                             ", got " + std::to_string(vector.size()));
    }
}

void SageVDB::ensure_consistent_metadata(const std::vector<Vector>& vectors,
                                       const std::vector<Metadata>& metadata) const {
    if (vectors.size() != metadata.size()) {
        throw SageVDBException("Vectors and metadata must have the same size");
    }
}

// Factory functions
std::unique_ptr<SageVDB> create_database(Dimension dimension,
                                       IndexType index_type,
                                       DistanceMetric metric) {
    DatabaseConfig config(dimension);
    config.index_type = index_type;
    config.metric = metric;
    return std::make_unique<SageVDB>(config);
}

std::unique_ptr<SageVDB> create_database(const DatabaseConfig& config) {
    return std::make_unique<SageVDB>(config);
}

// Utility functions
std::string index_type_to_string(IndexType type) {
    switch (type) {
        case IndexType::FLAT: return "FLAT";
        case IndexType::IVF_FLAT: return "IVF_FLAT";
        case IndexType::IVF_PQ: return "IVF_PQ";
        case IndexType::HNSW: return "HNSW";
        case IndexType::AUTO: return "AUTO";
        default: return "UNKNOWN";
    }
}

IndexType string_to_index_type(const std::string& str) {
    if (str == "FLAT") return IndexType::FLAT;
    if (str == "IVF_FLAT") return IndexType::IVF_FLAT;
    if (str == "IVF_PQ") return IndexType::IVF_PQ;
    if (str == "HNSW") return IndexType::HNSW;
    if (str == "AUTO") return IndexType::AUTO;
    throw SageVDBException("Unknown index type: " + str);
}

std::string distance_metric_to_string(DistanceMetric metric) {
    switch (metric) {
        case DistanceMetric::L2: return "L2";
        case DistanceMetric::INNER_PRODUCT: return "INNER_PRODUCT";
        case DistanceMetric::COSINE: return "COSINE";
        default: return "UNKNOWN";
    }
}

DistanceMetric string_to_distance_metric(const std::string& str) {
    if (str == "L2") return DistanceMetric::L2;
    if (str == "INNER_PRODUCT") return DistanceMetric::INNER_PRODUCT;
    if (str == "COSINE") return DistanceMetric::COSINE;
    throw SageVDBException("Unknown distance metric: " + str);
}

} // namespace sage_vdb

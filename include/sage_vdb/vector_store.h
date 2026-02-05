#pragma once

#include "common.h"
#include "anns/anns_interface.h"
#include <shared_mutex>

namespace sage_vdb {

class VectorStore {
public:
    VectorStore(const DatabaseConfig& config);
    ~VectorStore();

    // Basic operations
    VectorId add_vector(const Vector& vector);
    bool remove_vector(VectorId id);
    bool update_vector(VectorId id, const Vector& vector);
    
    // Batch operations
    std::vector<VectorId> add_vectors(const std::vector<Vector>& vectors);
    
    // Search operations
    std::vector<QueryResult> search(const Vector& query, const SearchParams& params) const;
    std::vector<std::vector<QueryResult>> batch_search(
        const std::vector<Vector>& queries, const SearchParams& params) const;
    
    // Index management
    void build_index();
    void train_index(const std::vector<Vector>& training_data);
    bool is_trained() const;
    
    // Statistics
    size_t size() const;
    Dimension dimension() const;
    IndexType index_type() const;
    
    // Persistence
    void save(const std::string& filepath) const;
    void load(const std::string& filepath);
    
    // Configuration
    const DatabaseConfig& config() const { return config_; }
    
private:
    class Impl;
    std::unique_ptr<Impl> impl_;
    DatabaseConfig config_;
    mutable std::shared_mutex mutex_;  // Allow concurrent reads!
    
    // Helper methods
    void validate_vector(const Vector& vector) const;
    void ensure_trained() const;
};

} // namespace sage_vdb

#include "sage_vdb/multimodal_sage_vdb.h"
#include "sage_vdb/fusion_strategies.h"

namespace sage_vdb {

MultimodalSageVDB::MultimodalSageVDB(const MultimodalConfig& config) 
    : SageVDB(config.base_config), multimodal_config_(config) {
    
    if (!validate_multimodal_config()) {
        throw SageVDBException("Invalid multimodal configuration");
    }
    
    // Initialize the modality manager
    modality_manager_ = std::make_unique<ModalityManager>();
    
    // Initialize the fusion engine
    fusion_engine_ = std::make_unique<FusionEngine>();
    
    // Initialize metadata store
    multimodal_metadata_store_ = std::make_shared<MetadataStore>();
    
    // Register default fusion strategies
    register_default_fusion_strategies();
}

VectorId MultimodalSageVDB::add_multimodal(const MultimodalData& data) {
    validate_multimodal_data(data);
    
    // Process each modality to get embeddings
    std::unordered_map<ModalityType, Vector> modality_vectors;
    
    for (const auto& [type, modal_data] : data.modalities) {
        // Use the embedding directly from ModalData
        modality_vectors[type] = modal_data.embedding;
    }
    
    // Perform fusion
    auto fused_vector = perform_fusion(modality_vectors);
    
    // Add the fused vector to the base SageVDB
    return SageVDB::add(fused_vector, data.global_metadata);
}

VectorId MultimodalSageVDB::add_multimodal(
    const std::unordered_map<ModalityType, ModalData>& modalities, 
    const Metadata& global_metadata) {
    
    MultimodalData data;
    data.modalities = modalities;
    data.global_metadata = global_metadata;
    return add_multimodal(data);
}

std::vector<QueryResult> MultimodalSageVDB::search_multimodal(
    const std::unordered_map<ModalityType, ModalData>& query_modalities,
    const MultimodalSearchParams& params) const {
    
    // Build the query vector from multimodal data
    Vector query_vector = build_query_vector(query_modalities, params);
    
    // Convert to base search parameters
    SearchParams base_params;
    base_params.k = params.k;
    base_params.include_metadata = params.include_metadata;
    
    return SageVDB::search(query_vector, base_params);
}

void MultimodalSageVDB::validate_multimodal_data(const MultimodalData& data) const {
    if (data.modalities.empty()) {
        throw MultimodalException("Multimodal data must contain at least one modality");
    }
    
    if (data.modalities.size() > multimodal_config_.max_modalities_per_item) {
        throw MultimodalException("Too many modalities per item: " +
                                 std::to_string(data.modalities.size()));
    }
    
    // Validate each modality has non-empty embedding
    for (const auto& [type, modal_data] : data.modalities) {
        if (modal_data.embedding.empty()) {
            throw MultimodalException("Empty embedding for modality type: " +
                                     std::to_string(static_cast<int>(type)));
        }
    }
}

Vector MultimodalSageVDB::perform_fusion(
    const std::unordered_map<ModalityType, Vector>& modal_embeddings) const {
    
    return fusion_engine_->fuse_embeddings(modal_embeddings, multimodal_config_.default_fusion_params);
}

Vector MultimodalSageVDB::build_query_vector(
    const std::unordered_map<ModalityType, ModalData>& query_modalities,
    const MultimodalSearchParams& params) const {
    
    // Convert modal data to vectors using embeddings
    std::unordered_map<ModalityType, Vector> query_vectors;
    
    for (const auto& [type, modal_data] : query_modalities) {
        // Use the embedding directly from ModalData
        query_vectors[type] = modal_data.embedding;
    }
    
    // Use the query fusion params if they have a different strategy than default,
    // otherwise use the database's default fusion params
    const FusionParams& fusion_params = 
        (params.query_fusion_params.strategy != multimodal_config_.default_fusion_params.strategy) ?
        params.query_fusion_params : multimodal_config_.default_fusion_params;
    
    return fusion_engine_->fuse_embeddings(query_vectors, fusion_params);
}

void MultimodalSageVDB::register_modality_processor(ModalityType type,
                                                  std::shared_ptr<ModalityProcessor> processor) {
    modality_manager_->register_processor(type, processor);
}

void MultimodalSageVDB::register_fusion_strategy(FusionStrategy strategy,
                                               std::shared_ptr<FusionStrategyInterface> impl) {
    fusion_engine_->register_strategy(strategy, impl);
}

void MultimodalSageVDB::update_fusion_params(const FusionParams& params) {
    multimodal_config_.default_fusion_params = params;
}

const FusionParams& MultimodalSageVDB::get_fusion_params() const {
    return multimodal_config_.default_fusion_params;
}

std::vector<ModalityType> MultimodalSageVDB::get_supported_modalities() const {
    return modality_manager_->get_supported_modalities();
}

std::vector<FusionStrategy> MultimodalSageVDB::get_supported_fusion_strategies() const {
    return fusion_engine_->get_supported_strategies();
}

bool MultimodalSageVDB::validate_multimodal_config() const {
    // Validate max modalities per item
    if (multimodal_config_.max_modalities_per_item == 0) {
        return false;
    }
    
    if (multimodal_config_.base_config.dimension == 0) {
        return false;
    }
    
    // Additional validation can be added here
    return true;
}

void MultimodalSageVDB::register_default_fusion_strategies() {
    // Register concatenation fusion
    fusion_engine_->register_strategy(
        FusionStrategy::CONCATENATION,
        FusionStrategyFactory::create_concatenation_fusion()
    );
    
    // Register weighted average fusion
    fusion_engine_->register_strategy(
        FusionStrategy::WEIGHTED_AVERAGE,
        FusionStrategyFactory::create_weighted_average_fusion()
    );
    
    // Register attention-based fusion
    fusion_engine_->register_strategy(
        FusionStrategy::ATTENTION_BASED,
        FusionStrategyFactory::create_attention_based_fusion()
    );
    
    // Register tensor fusion
    fusion_engine_->register_strategy(
        FusionStrategy::TENSOR_FUSION,
        FusionStrategyFactory::create_tensor_fusion()
    );
}

// Factory methods for creating specialized multimodal databases
std::unique_ptr<MultimodalSageVDB> MultimodalSageVDBFactory::create_text_image_db(
    const DatabaseConfig& base_config) {
    
    MultimodalConfig config;
    config.base_config = base_config;
    config.default_fusion_params.strategy = FusionStrategy::WEIGHTED_AVERAGE;
    config.default_fusion_params.modality_weights[ModalityType::TEXT] = 0.6f;
    config.default_fusion_params.modality_weights[ModalityType::IMAGE] = 0.4f;
    config.default_fusion_params.target_dimension = base_config.dimension;  // 确保目标维度匹配
    config.max_modalities_per_item = 2;
    
    return std::make_unique<MultimodalSageVDB>(config);
}

std::unique_ptr<MultimodalSageVDB> MultimodalSageVDBFactory::create_audio_visual_db(
    const DatabaseConfig& base_config) {
    
    MultimodalConfig config;
    config.base_config = base_config;
    config.default_fusion_params.strategy = FusionStrategy::ATTENTION_BASED;
    config.default_fusion_params.modality_weights[ModalityType::AUDIO] = 0.5f;
    config.default_fusion_params.modality_weights[ModalityType::VIDEO] = 0.5f;
    config.max_modalities_per_item = 2;
    
    return std::make_unique<MultimodalSageVDB>(config);
}

} // namespace sage_vdb
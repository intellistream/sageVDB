#pragma once

#include "common.h"
#include "sage_vdb.h"
#include "multimodal_fusion.h"

namespace sage_vdb {

// 多模态SageVDB扩展类
class MultimodalSageVDB : public SageVDB {
public:
    explicit MultimodalSageVDB(const MultimodalConfig& config);
    ~MultimodalSageVDB() = default;

    VectorId add_multimodal(const MultimodalData& data);
    VectorId add_multimodal(const std::unordered_map<ModalityType, ModalData>& modalities,
                           const Metadata& global_metadata = {});

    std::vector<QueryResult> search_multimodal(
        const std::unordered_map<ModalityType, ModalData>& query_modalities,
        const MultimodalSearchParams& params) const;

    void register_modality_processor(ModalityType type,
                                   std::shared_ptr<ModalityProcessor> processor);
    void register_fusion_strategy(FusionStrategy strategy,
                                 std::shared_ptr<FusionStrategyInterface> impl);

    void update_fusion_params(const FusionParams& params);
    const FusionParams& get_fusion_params() const;

    std::vector<ModalityType> get_supported_modalities() const;
    std::vector<FusionStrategy> get_supported_fusion_strategies() const;

    bool validate_multimodal_config() const;

private:
    MultimodalConfig multimodal_config_;
    std::unique_ptr<ModalityManager> modality_manager_;
    std::unique_ptr<FusionEngine> fusion_engine_;
    std::shared_ptr<MetadataStore> multimodal_metadata_store_;

    void validate_multimodal_data(const MultimodalData& data) const;
    Vector perform_fusion(const std::unordered_map<ModalityType, Vector>& modal_embeddings) const;
    Vector build_query_vector(const std::unordered_map<ModalityType, ModalData>& query_modalities,
                             const MultimodalSearchParams& params) const;

    void register_default_fusion_strategies();
    void register_default_modality_processors();
};

// ========== 工厂类 ==========

class MultimodalSageVDBFactory {
public:
    // 创建预配置的多模态数据库实例
    static std::unique_ptr<MultimodalSageVDB> create_text_image_db(
        const DatabaseConfig& base_config);
    
    static std::unique_ptr<MultimodalSageVDB> create_audio_visual_db(
        const DatabaseConfig& base_config);
    
    static std::unique_ptr<MultimodalSageVDB> create_full_multimodal_db(
        const DatabaseConfig& base_config);
    
    // 创建自定义配置的实例
    static std::unique_ptr<MultimodalSageVDB> create_custom_db(
        const MultimodalConfig& config);
};

} // namespace sage_vdb
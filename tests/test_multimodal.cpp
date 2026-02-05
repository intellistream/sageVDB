#include "sage_vdb/multimodal_sage_vdb.h"
#include "sage_vdb/fusion_strategies.h"
#include <cassert>
#include <iostream>
#include <numeric>

using namespace sage_vdb;

// 测试基本的多模态功能
void test_basic_multimodal() {
    std::cout << "测试基本多模态功能..." << std::endl;
    
    DatabaseConfig config;
    config.dimension = 128;
    config.index_type = IndexType::FLAT;
    
    MultimodalConfig multimodal_config;
    multimodal_config.base_config = config;
    multimodal_config.max_modalities_per_item = 3;
    // 设置fusion参数，确保融合后的向量维度与数据库配置一致
    multimodal_config.default_fusion_params.target_dimension = config.dimension;
    multimodal_config.default_fusion_params.strategy = FusionStrategy::CONCATENATION;
    
    MultimodalSageVDB db(multimodal_config);
    
    // 创建测试数据
    Vector text_embedding(100);
    std::iota(text_embedding.begin(), text_embedding.end(), 1.0f);
    
    Vector image_embedding(200);
    std::iota(image_embedding.begin(), image_embedding.end(), 2.0f);
    
    ModalData text_modal(ModalityType::TEXT, text_embedding);
    ModalData image_modal(ModalityType::IMAGE, image_embedding);
    
    std::unordered_map<ModalityType, ModalData> modalities;
    modalities[ModalityType::TEXT] = text_modal;
    modalities[ModalityType::IMAGE] = image_modal;
    
    VectorId id = db.add_multimodal(modalities);
    assert(id > 0);
    
    std::cout << "✓ 基本多模态功能测试通过" << std::endl;
}

// 测试融合策略
void test_fusion_strategies() {
    std::cout << "测试融合策略..." << std::endl;
    
    // 创建测试数据
    std::unordered_map<ModalityType, Vector> modal_embeddings;
    modal_embeddings[ModalityType::TEXT] = Vector{1.0f, 2.0f, 3.0f};
    modal_embeddings[ModalityType::IMAGE] = Vector{4.0f, 5.0f, 6.0f};
    
    FusionParams params;
    
    // 测试拼接融合
    {
        ConcatenationFusion fusion;
        Vector result = fusion.fuse(modal_embeddings, params);
        assert(result.size() == 6);
        std::cout << "✓ 拼接融合测试通过" << std::endl;
    }
    
    // 测试加权平均融合
    {
        WeightedAverageFusion fusion;
        params.target_dimension = 3;
        Vector result = fusion.fuse(modal_embeddings, params);
        assert(result.size() == 3);
        std::cout << "✓ 加权平均融合测试通过" << std::endl;
    }
    
    // 测试注意力融合
    {
        AttentionBasedFusion fusion;
        Vector result = fusion.fuse(modal_embeddings, params);
        assert(result.size() == 3);
        std::cout << "✓ 注意力融合测试通过" << std::endl;
    }
    
    std::cout << "✓ 所有融合策略测试通过" << std::endl;
}

// 测试工具函数
void test_fusion_utils() {
    std::cout << "测试融合工具函数..." << std::endl;
    
    Vector v1{1.0f, 2.0f, 3.0f};
    Vector v2{4.0f, 5.0f, 6.0f};
    
    // 测试余弦相似度
    float sim = fusion_utils::cosine_similarity(v1, v2);
    assert(sim > 0.9f && sim <= 1.0f);
    
    // 测试维度对齐
    Vector aligned = fusion_utils::align_dimension(v1, 5);
    assert(aligned.size() == 5);
    
    // 测试平均池化
    std::vector<Vector> vectors = {v1, v2};
    Vector pooled = fusion_utils::avg_pooling(vectors);
    assert(pooled.size() == 3);
    assert(std::abs(pooled[0] - 2.5f) < 0.01f);
    
    std::cout << "✓ 融合工具函数测试通过" << std::endl;
}

// 测试工厂类
void test_factory() {
    std::cout << "测试工厂类..." << std::endl;
    
    DatabaseConfig config;
    config.dimension = 256;
    config.index_type = IndexType::FLAT;
    
    // 测试文本-图像数据库
    auto text_image_db = MultimodalSageVDBFactory::create_text_image_db(config);
    assert(text_image_db != nullptr);
    assert(text_image_db->validate_multimodal_config());
    
    // 测试音视频数据库
    auto audio_visual_db = MultimodalSageVDBFactory::create_audio_visual_db(config);
    assert(audio_visual_db != nullptr);
    assert(audio_visual_db->validate_multimodal_config());
    
    std::cout << "✓ 工厂类测试通过" << std::endl;
}

// 测试自定义融合策略注册
void test_custom_strategy() {
    std::cout << "测试自定义融合策略..." << std::endl;
    
    class TestCustomFusion : public FusionStrategyInterface {
    public:
        Vector fuse(const std::unordered_map<ModalityType, Vector>& modal_embeddings,
                   const FusionParams& params) override {
            // 简单的求和融合
            Vector result;
            for (const auto& [type, embedding] : modal_embeddings) {
                if (result.empty()) {
                    result = embedding;
                } else {
                    for (size_t i = 0; i < std::min(result.size(), embedding.size()); ++i) {
                        result[i] += embedding[i];
                    }
                }
            }
            
            // 如果指定了目标维度，进行维度调整
            if (params.target_dimension > 0 && result.size() != params.target_dimension) {
                Vector aligned_result(params.target_dimension, 0.0f);
                if (result.size() <= params.target_dimension) {
                    // 扩展：复制并用零填充
                    std::copy(result.begin(), result.end(), aligned_result.begin());
                } else {
                    // 收缩：截断
                    std::copy(result.begin(), result.begin() + params.target_dimension, aligned_result.begin());
                }
                result = aligned_result;
            }
            
            return result;
        }
        
        FusionStrategy get_strategy_type() const override {
            return FusionStrategy::CUSTOM;
        }
    };
    
    DatabaseConfig config;
    config.dimension = 128;
    config.index_type = IndexType::FLAT;
    
    MultimodalConfig multimodal_config;
    multimodal_config.base_config = config;
    // 设置target_dimension以确保自定义融合策略输出正确维度
    multimodal_config.default_fusion_params.target_dimension = config.dimension;
    
    MultimodalSageVDB db(multimodal_config);
    
    // 注册自定义策略
    db.register_fusion_strategy(FusionStrategy::CUSTOM, 
                               std::make_shared<TestCustomFusion>());
    
    // 测试自定义策略
    FusionParams params;
    params.strategy = FusionStrategy::CUSTOM;
    params.target_dimension = config.dimension;  // 确保设置目标维度
    db.update_fusion_params(params);
    
    Vector text_embedding{1.0f, 2.0f};
    Vector image_embedding{3.0f, 4.0f};
    
    std::unordered_map<ModalityType, ModalData> modalities;
    modalities[ModalityType::TEXT] = ModalData(ModalityType::TEXT, text_embedding);
    modalities[ModalityType::IMAGE] = ModalData(ModalityType::IMAGE, image_embedding);
    
    VectorId id = db.add_multimodal(modalities);
    assert(id > 0);
    
    std::cout << "✓ 自定义融合策略测试通过" << std::endl;
}

int main() {
    std::cout << "=== SageVDB 多模态融合模块单元测试 ===" << std::endl;
    
    try {
        test_basic_multimodal();
        test_fusion_strategies();
        test_fusion_utils();
        test_factory();
        test_custom_strategy();
        
        std::cout << "\n✅ 所有单元测试通过！" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "❌ 测试失败: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}
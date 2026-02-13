#pragma once

#include "multimodal_fusion.h"
#include <numeric>
#include <cmath>

namespace sage_vdb {

// ========== 向量拼接融合策略 ==========
class ConcatenationFusion : public FusionStrategyInterface {
public:
    Vector fuse(const std::unordered_map<ModalityType, Vector>& modal_embeddings,
               const FusionParams& params) override;
    
    FusionStrategy get_strategy_type() const override {
        return FusionStrategy::CONCATENATION;
    }
};

// ========== 加权平均融合策略 ==========
class WeightedAverageFusion : public FusionStrategyInterface {
public:
    Vector fuse(const std::unordered_map<ModalityType, Vector>& modal_embeddings,
               const FusionParams& params) override;
    
    FusionStrategy get_strategy_type() const override {
        return FusionStrategy::WEIGHTED_AVERAGE;
    }

private:
    Vector normalize_weights(const std::unordered_map<ModalityType, float>& weights,
                           const std::vector<ModalityType>& present_modalities) const;
};

// ========== 注意力机制融合策略 ==========
class AttentionBasedFusion : public FusionStrategyInterface {
public:
    AttentionBasedFusion();
    
    Vector fuse(const std::unordered_map<ModalityType, Vector>& modal_embeddings,
               const FusionParams& params) override;
    
    FusionStrategy get_strategy_type() const override {
        return FusionStrategy::ATTENTION_BASED;
    }

private:
    struct AttentionWeights {
        std::unordered_map<ModalityType, float> weights;
        float total_weight;
    };
    
    AttentionWeights compute_attention_weights(
        const std::unordered_map<ModalityType, Vector>& modal_embeddings) const;
    
    float compute_modality_attention(const Vector& embedding, 
                                   const Vector& context) const;
    
    Vector compute_context_vector(
        const std::unordered_map<ModalityType, Vector>& modal_embeddings) const;
};

// ========== 张量融合策略 ==========
class TensorFusion : public FusionStrategyInterface {
public:
    TensorFusion(uint32_t target_dim = 512);
    
    Vector fuse(const std::unordered_map<ModalityType, Vector>& modal_embeddings,
               const FusionParams& params) override;
    
    FusionStrategy get_strategy_type() const override {
        return FusionStrategy::TENSOR_FUSION;
    }

private:
    uint32_t target_dimension_;
    
    Vector compute_tensor_product(const Vector& v1, const Vector& v2) const;
    Vector reduce_dimension(const Vector& tensor_product, uint32_t target_dim) const;
};

// ========== 双线性池化融合策略 ==========
class BilinearPoolingFusion : public FusionStrategyInterface {
public:
    BilinearPoolingFusion(uint32_t target_dim = 512);
    
    Vector fuse(const std::unordered_map<ModalityType, Vector>& modal_embeddings,
               const FusionParams& params) override;
    
    FusionStrategy get_strategy_type() const override {
        return FusionStrategy::BILINEAR_POOLING;
    }

private:
    uint32_t target_dimension_;
    
    Vector bilinear_pool(const Vector& v1, const Vector& v2) const;
    Vector compact_bilinear_pool(const Vector& v1, const Vector& v2) const;
};

// ========== 跨模态Transformer融合策略 ==========
class CrossModalTransformerFusion : public FusionStrategyInterface {
public:
    struct TransformerConfig {
        uint32_t hidden_dim;
        uint32_t num_heads;
        uint32_t num_layers;
        float dropout_rate;
        
        TransformerConfig() : hidden_dim(512), num_heads(8), num_layers(2), dropout_rate(0.1f) {}
    };
    
    CrossModalTransformerFusion(const TransformerConfig& config = TransformerConfig());
    
    Vector fuse(const std::unordered_map<ModalityType, Vector>& modal_embeddings,
               const FusionParams& params) override;
    
    FusionStrategy get_strategy_type() const override {
        return FusionStrategy::CROSS_MODAL_TRANSFORMER;
    }

private:
    TransformerConfig config_;
    
    struct MultiHeadAttentionOutput {
        Vector output;
        std::unordered_map<ModalityType, Vector> attention_weights;
    };
    
    MultiHeadAttentionOutput multi_head_attention(
        const std::unordered_map<ModalityType, Vector>& modal_embeddings) const;
    
    Vector feed_forward(const Vector& input) const;
    Vector layer_norm(const Vector& input) const;
};

// ========== 融合策略工厂 ==========
class FusionStrategyFactory {
public:
    static std::shared_ptr<FusionStrategyInterface> create_strategy(
        FusionStrategy strategy_type);
    
    static std::shared_ptr<FusionStrategyInterface> create_concatenation_fusion();
    static std::shared_ptr<FusionStrategyInterface> create_weighted_average_fusion();
    static std::shared_ptr<FusionStrategyInterface> create_attention_based_fusion();
    static std::shared_ptr<FusionStrategyInterface> create_tensor_fusion(uint32_t target_dim = 512);
    static std::shared_ptr<FusionStrategyInterface> create_bilinear_pooling_fusion(uint32_t target_dim = 512);
    static std::shared_ptr<FusionStrategyInterface> create_cross_modal_transformer_fusion(
        const CrossModalTransformerFusion::TransformerConfig& config = CrossModalTransformerFusion::TransformerConfig());
    
    // 注册自定义融合策略
    static void register_custom_strategy(
        const std::string& name,
        std::function<std::shared_ptr<FusionStrategyInterface>()> factory_func);
    
    static std::shared_ptr<FusionStrategyInterface> create_custom_strategy(
        const std::string& name);

private:
    static std::unordered_map<std::string, std::function<std::shared_ptr<FusionStrategyInterface>()>> custom_strategies_;
};

// ========== 辅助函数 ==========

namespace fusion_utils {
    // 向量归一化
    Vector normalize_vector(const Vector& vec);
    
    // 向量相似度计算
    float cosine_similarity(const Vector& v1, const Vector& v2);
    float euclidean_distance(const Vector& v1, const Vector& v2);
    
    // 维度对齐
    Vector align_dimension(const Vector& vec, uint32_t target_dim);
    
    // 向量池化操作
    Vector max_pooling(const std::vector<Vector>& vectors);
    Vector avg_pooling(const std::vector<Vector>& vectors);
    Vector sum_pooling(const std::vector<Vector>& vectors);
    
    // 激活函数
    Vector relu(const Vector& vec);
    Vector sigmoid(const Vector& vec);
    Vector tanh_activation(const Vector& vec);
    
    // 随机投影降维
    Vector random_projection(const Vector& vec, uint32_t target_dim, uint64_t seed = 42);
    
    // PCA降维（简化版本）
    Vector pca_projection(const Vector& vec, const std::vector<Vector>& principal_components);
}

} // namespace sage_vdb
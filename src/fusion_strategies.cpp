#include "sage_vdb/fusion_strategies.h"
#include <algorithm>
#include <random>
#include <stdexcept>
#include <limits>

namespace sage_vdb {

// ========== ConcatenationFusion 实现 ==========
Vector ConcatenationFusion::fuse(const std::unordered_map<ModalityType, Vector>& modal_embeddings,
                                 const FusionParams& params) {
    Vector result;
    
    // 确定拼接顺序（按模态类型排序以保证一致性）
    std::vector<ModalityType> sorted_types;
    for (const auto& [type, embedding] : modal_embeddings) {
        sorted_types.push_back(type);
    }
    std::sort(sorted_types.begin(), sorted_types.end());
    
    // 拼接向量
    for (auto type : sorted_types) {
        const auto& embedding = modal_embeddings.at(type);
        result.insert(result.end(), embedding.begin(), embedding.end());
    }
    
    // 如果指定了目标维度，进行维度调整
    if (params.target_dimension > 0 && result.size() != params.target_dimension) {
        result = fusion_utils::align_dimension(result, params.target_dimension);
    }
    
    return result;
}

// ========== WeightedAverageFusion 实现 ==========
Vector WeightedAverageFusion::fuse(const std::unordered_map<ModalityType, Vector>& modal_embeddings,
                                  const FusionParams& params) {
    if (modal_embeddings.empty()) {
        return Vector();
    }
    
    // 获取第一个嵌入的维度作为目标维度
    uint32_t target_dim = modal_embeddings.begin()->second.size();
    if (params.target_dimension > 0) {
        target_dim = params.target_dimension;
    }
    
    Vector result(target_dim, 0.0f);
    float total_weight = 0.0f;
    
    for (const auto& [type, embedding] : modal_embeddings) {
        // 获取权重
        float weight = 1.0f;
        auto weight_it = params.modality_weights.find(type);
        if (weight_it != params.modality_weights.end()) {
            weight = weight_it->second;
        }
        
        // 维度对齐
        Vector aligned_embedding = fusion_utils::align_dimension(embedding, target_dim);
        
        // 加权累加
        for (size_t i = 0; i < target_dim; ++i) {
            result[i] += weight * aligned_embedding[i];
        }
        
        total_weight += weight;
    }
    
    // 归一化
    if (total_weight > 0.0f) {
        for (float& val : result) {
            val /= total_weight;
        }
    }
    
    return result;
}

// ========== AttentionBasedFusion 实现 ==========
AttentionBasedFusion::AttentionBasedFusion() {}

Vector AttentionBasedFusion::fuse(const std::unordered_map<ModalityType, Vector>& modal_embeddings,
                                 const FusionParams& params) {
    if (modal_embeddings.empty()) {
        return Vector();
    }
    
    // 计算注意力权重
    auto attention_weights = compute_attention_weights(modal_embeddings);
    
    // 获取目标维度
    uint32_t target_dim = modal_embeddings.begin()->second.size();
    if (params.target_dimension > 0) {
        target_dim = params.target_dimension;
    }
    
    Vector result(target_dim, 0.0f);
    
    for (const auto& [type, embedding] : modal_embeddings) {
        float weight = attention_weights.weights.at(type);
        Vector aligned_embedding = fusion_utils::align_dimension(embedding, target_dim);
        
        for (size_t i = 0; i < target_dim; ++i) {
            result[i] += weight * aligned_embedding[i];
        }
    }
    
    return result;
}

AttentionBasedFusion::AttentionWeights AttentionBasedFusion::compute_attention_weights(
    const std::unordered_map<ModalityType, Vector>& modal_embeddings) const {
    
    AttentionWeights weights;
    Vector context = compute_context_vector(modal_embeddings);
    
    float total_attention = 0.0f;
    for (const auto& [type, embedding] : modal_embeddings) {
        float attention = compute_modality_attention(embedding, context);
        weights.weights[type] = attention;
        total_attention += attention;
    }
    
    // 归一化权重
    if (total_attention > 0.0f) {
        for (auto& [type, weight] : weights.weights) {
            weight /= total_attention;
        }
    }
    
    weights.total_weight = total_attention;
    return weights;
}

float AttentionBasedFusion::compute_modality_attention(const Vector& embedding, 
                                                      const Vector& context) const {
    // 简单的点积注意力
    float attention = fusion_utils::cosine_similarity(embedding, context);
    return std::max(0.0f, attention); // ReLU激活
}

Vector AttentionBasedFusion::compute_context_vector(
    const std::unordered_map<ModalityType, Vector>& modal_embeddings) const {
    
    // 计算所有嵌入的平均向量作为上下文
    std::vector<Vector> embeddings;
    for (const auto& [type, embedding] : modal_embeddings) {
        embeddings.push_back(embedding);
    }
    
    return fusion_utils::avg_pooling(embeddings);
}

// ========== TensorFusion 实现 ==========
TensorFusion::TensorFusion(uint32_t target_dim) : target_dimension_(target_dim) {}

Vector TensorFusion::fuse(const std::unordered_map<ModalityType, Vector>& modal_embeddings,
                         const FusionParams& params) {
    if (modal_embeddings.empty()) {
        return Vector();
    }
    
    if (modal_embeddings.size() == 1) {
        // 单个模态直接返回
        const auto& embedding = modal_embeddings.begin()->second;
        return fusion_utils::align_dimension(embedding, target_dimension_);
    }
    
    // 获取所有嵌入向量
    std::vector<Vector> embeddings;
    for (const auto& [type, embedding] : modal_embeddings) {
        embeddings.push_back(embedding);
    }
    
    // 计算张量乘积（这里简化为两两乘积）
    Vector result = embeddings[0];
    for (size_t i = 1; i < embeddings.size(); ++i) {
        result = compute_tensor_product(result, embeddings[i]);
    }
    
    // 降维到目标维度
    uint32_t target_dim = params.target_dimension > 0 ? params.target_dimension : target_dimension_;
    return reduce_dimension(result, target_dim);
}

Vector TensorFusion::compute_tensor_product(const Vector& v1, const Vector& v2) const {
    Vector result;
    result.reserve(v1.size() * v2.size());
    
    for (float val1 : v1) {
        for (float val2 : v2) {
            result.push_back(val1 * val2);
        }
    }
    
    return result;
}

Vector TensorFusion::reduce_dimension(const Vector& tensor_product, uint32_t target_dim) const {
    if (tensor_product.size() <= target_dim) {
        return tensor_product;
    }
    
    // 使用随机投影降维
    return fusion_utils::random_projection(tensor_product, target_dim);
}

// ========== BilinearPoolingFusion 实现 ==========
BilinearPoolingFusion::BilinearPoolingFusion(uint32_t target_dim) : target_dimension_(target_dim) {}

Vector BilinearPoolingFusion::fuse(const std::unordered_map<ModalityType, Vector>& modal_embeddings,
                                  const FusionParams& params) {
    if (modal_embeddings.empty()) {
        return Vector();
    }
    
    if (modal_embeddings.size() == 1) {
        const auto& embedding = modal_embeddings.begin()->second;
        return fusion_utils::align_dimension(embedding, target_dimension_);
    }
    
    // 获取所有嵌入向量
    std::vector<Vector> embeddings;
    for (const auto& [type, embedding] : modal_embeddings) {
        embeddings.push_back(embedding);
    }
    
    // 计算双线性池化（两两池化）
    Vector result = embeddings[0];
    for (size_t i = 1; i < embeddings.size(); ++i) {
        result = bilinear_pool(result, embeddings[i]);
    }
    
    uint32_t target_dim = params.target_dimension > 0 ? params.target_dimension : target_dimension_;
    return fusion_utils::align_dimension(result, target_dim);
}

Vector BilinearPoolingFusion::bilinear_pool(const Vector& v1, const Vector& v2) const {
    // 简化的双线性池化：element-wise乘积后求和池化
    Vector aligned_v1 = v1;
    Vector aligned_v2 = v2;
    
    // 对齐维度
    size_t min_dim = std::min(v1.size(), v2.size());
    aligned_v1.resize(min_dim);
    aligned_v2.resize(min_dim);
    
    Vector result(min_dim);
    for (size_t i = 0; i < min_dim; ++i) {
        result[i] = aligned_v1[i] * aligned_v2[i];
    }
    
    return result;
}

Vector BilinearPoolingFusion::compact_bilinear_pool(const Vector& v1, const Vector& v2) const {
    // 紧凑双线性池化的简化实现
    return bilinear_pool(v1, v2);
}

// ========== FusionStrategyFactory 实现 ==========
std::unordered_map<std::string, std::function<std::shared_ptr<FusionStrategyInterface>()>> 
FusionStrategyFactory::custom_strategies_;

std::shared_ptr<FusionStrategyInterface> FusionStrategyFactory::create_strategy(
    FusionStrategy strategy_type) {
    
    switch (strategy_type) {
        case FusionStrategy::CONCATENATION:
            return create_concatenation_fusion();
        case FusionStrategy::WEIGHTED_AVERAGE:
            return create_weighted_average_fusion();
        case FusionStrategy::ATTENTION_BASED:
            return create_attention_based_fusion();
        case FusionStrategy::TENSOR_FUSION:
            return create_tensor_fusion();
        case FusionStrategy::BILINEAR_POOLING:
            return create_bilinear_pooling_fusion();
        default:
            throw std::invalid_argument("Unsupported fusion strategy");
    }
}

std::shared_ptr<FusionStrategyInterface> FusionStrategyFactory::create_concatenation_fusion() {
    return std::make_shared<ConcatenationFusion>();
}

std::shared_ptr<FusionStrategyInterface> FusionStrategyFactory::create_weighted_average_fusion() {
    return std::make_shared<WeightedAverageFusion>();
}

std::shared_ptr<FusionStrategyInterface> FusionStrategyFactory::create_attention_based_fusion() {
    return std::make_shared<AttentionBasedFusion>();
}

std::shared_ptr<FusionStrategyInterface> FusionStrategyFactory::create_tensor_fusion(uint32_t target_dim) {
    return std::make_shared<TensorFusion>(target_dim);
}

std::shared_ptr<FusionStrategyInterface> FusionStrategyFactory::create_bilinear_pooling_fusion(uint32_t target_dim) {
    return std::make_shared<BilinearPoolingFusion>(target_dim);
}

void FusionStrategyFactory::register_custom_strategy(
    const std::string& name,
    std::function<std::shared_ptr<FusionStrategyInterface>()> factory_func) {
    custom_strategies_[name] = factory_func;
}

std::shared_ptr<FusionStrategyInterface> FusionStrategyFactory::create_custom_strategy(
    const std::string& name) {
    auto it = custom_strategies_.find(name);
    if (it != custom_strategies_.end()) {
        return it->second();
    }
    throw std::invalid_argument("Unknown custom strategy: " + name);
}

// ========== fusion_utils 实现 ==========
namespace fusion_utils {

Vector normalize_vector(const Vector& vec) {
    Vector result = vec;
    float norm = 0.0f;
    for (float val : vec) {
        norm += val * val;
    }
    norm = std::sqrt(norm);
    
    if (norm > 0.0f) {
        for (float& val : result) {
            val /= norm;
        }
    }
    
    return result;
}

float cosine_similarity(const Vector& v1, const Vector& v2) {
    if (v1.size() != v2.size()) {
        return 0.0f;
    }
    
    float dot_product = 0.0f;
    float norm1 = 0.0f;
    float norm2 = 0.0f;
    
    for (size_t i = 0; i < v1.size(); ++i) {
        dot_product += v1[i] * v2[i];
        norm1 += v1[i] * v1[i];
        norm2 += v2[i] * v2[i];
    }
    
    float denominator = std::sqrt(norm1 * norm2);
    return denominator > 0.0f ? dot_product / denominator : 0.0f;
}

float euclidean_distance(const Vector& v1, const Vector& v2) {
    if (v1.size() != v2.size()) {
        return std::numeric_limits<float>::max();
    }
    
    float sum = 0.0f;
    for (size_t i = 0; i < v1.size(); ++i) {
        float diff = v1[i] - v2[i];
        sum += diff * diff;
    }
    
    return std::sqrt(sum);
}

Vector align_dimension(const Vector& vec, uint32_t target_dim) {
    if (vec.size() == target_dim) {
        return vec;
    }
    
    Vector result(target_dim, 0.0f);
    
    if (vec.size() < target_dim) {
        // 扩展：复制并用零填充
        std::copy(vec.begin(), vec.end(), result.begin());
    } else {
        // 收缩：截断或平均池化
        float scale = static_cast<float>(vec.size()) / target_dim;
        for (uint32_t i = 0; i < target_dim; ++i) {
            size_t start_idx = static_cast<size_t>(i * scale);
            size_t end_idx = static_cast<size_t>((i + 1) * scale);
            end_idx = std::min(end_idx, vec.size());
            
            float sum = 0.0f;
            for (size_t j = start_idx; j < end_idx; ++j) {
                sum += vec[j];
            }
            result[i] = sum / (end_idx - start_idx);
        }
    }
    
    return result;
}

Vector avg_pooling(const std::vector<Vector>& vectors) {
    if (vectors.empty()) {
        return Vector();
    }
    
    Vector result(vectors[0].size(), 0.0f);
    
    for (const auto& vec : vectors) {
        for (size_t i = 0; i < std::min(result.size(), vec.size()); ++i) {
            result[i] += vec[i];
        }
    }
    
    for (float& val : result) {
        val /= vectors.size();
    }
    
    return result;
}

Vector random_projection(const Vector& vec, uint32_t target_dim, uint64_t seed) {
    std::mt19937 gen(seed);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    
    Vector result(target_dim, 0.0f);
    
    for (uint32_t i = 0; i < target_dim; ++i) {
        for (size_t j = 0; j < vec.size(); ++j) {
            result[i] += vec[j] * dist(gen);
        }
        result[i] /= std::sqrt(vec.size());
    }
    
    return result;
}

} // namespace fusion_utils

} // namespace sage_vdb
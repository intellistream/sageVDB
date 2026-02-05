#pragma once

#include "common.h"
#include <unordered_map>
#include <functional>
#include <memory>

namespace sage_vdb {

// 模态类型枚举
enum class ModalityType {
    TEXT,           // 文本
    IMAGE,          // 图像
    AUDIO,          // 音频
    VIDEO,          // 视频
    TABULAR,        // 表格数据
    GRAPH,          // 图结构
    TIME_SERIES,    // 时间序列
    CUSTOM          // 自定义模态
};

// 模态数据结构
struct ModalData {
    ModalityType type;
    Vector embedding;           // 该模态的嵌入向量
    Metadata metadata;          // 模态相关的元数据
    std::vector<uint8_t> raw_data;  // 原始数据（可选）
    
    // 默认构造函数
    ModalData() : type(ModalityType::TEXT) {}
    
    // 带参数的构造函数
    ModalData(ModalityType t, const Vector& emb, const Metadata& meta = {})
        : type(t), embedding(emb), metadata(meta) {}
};

// 多模态数据包
struct MultimodalData {
    VectorId id;
    std::unordered_map<ModalityType, ModalData> modalities;
    Vector fused_embedding;     // 融合后的嵌入向量
    Metadata global_metadata;   // 全局元数据
    
    MultimodalData() : id(0) {}
    MultimodalData(VectorId id_) : id(id_) {}
};

// 融合策略枚举
enum class FusionStrategy {
    CONCATENATION,      // 向量拼接
    WEIGHTED_AVERAGE,   // 加权平均
    ATTENTION_BASED,    // 注意力机制融合
    CROSS_MODAL_TRANSFORMER,  // 跨模态Transformer
    TENSOR_FUSION,      // 张量融合
    BILINEAR_POOLING,   // 双线性池化
    CUSTOM              // 自定义融合策略
};

// 融合参数
struct FusionParams {
    FusionStrategy strategy = FusionStrategy::WEIGHTED_AVERAGE;
    std::unordered_map<ModalityType, float> modality_weights;  // 模态权重
    uint32_t target_dimension = 0;  // 目标融合向量维度
    std::map<std::string, float> custom_params;  // 自定义参数
    
    FusionParams() {
        // 默认权重
        modality_weights[ModalityType::TEXT] = 0.4f;
        modality_weights[ModalityType::IMAGE] = 0.3f;
        modality_weights[ModalityType::AUDIO] = 0.2f;
        modality_weights[ModalityType::VIDEO] = 0.1f;
    }
};

// 模态处理器接口
class ModalityProcessor {
public:
    virtual ~ModalityProcessor() = default;
    virtual Vector process(const std::vector<uint8_t>& raw_data) = 0;
    virtual bool validate(const std::vector<uint8_t>& raw_data) const = 0;
    virtual ModalityType get_type() const = 0;
};

// 融合策略接口
class FusionStrategyInterface {
public:
    virtual ~FusionStrategyInterface() = default;
    virtual Vector fuse(const std::unordered_map<ModalityType, Vector>& modal_embeddings,
                       const FusionParams& params) = 0;
    virtual FusionStrategy get_strategy_type() const = 0;
};

// 模态管理器
class ModalityManager {
public:
    ModalityManager() = default;
    ~ModalityManager() = default;
    
    // 注册模态处理器
    void register_processor(ModalityType type, 
                          std::shared_ptr<ModalityProcessor> processor);
    
    // 处理原始数据为嵌入向量
    Vector process_modality(ModalityType type, 
                           const std::vector<uint8_t>& raw_data);
    
    // 验证模态数据
    bool validate_modality(ModalityType type, 
                          const std::vector<uint8_t>& raw_data) const;
    
    // 获取支持的模态类型
    std::vector<ModalityType> get_supported_modalities() const;

private:
    std::unordered_map<ModalityType, std::shared_ptr<ModalityProcessor>> processors_;
};

// 融合引擎
class FusionEngine {
public:
    FusionEngine() = default;
    ~FusionEngine() = default;
    
    // 注册融合策略
    void register_strategy(FusionStrategy strategy, 
                          std::shared_ptr<FusionStrategyInterface> impl);
    
    // 执行融合
    Vector fuse_embeddings(const std::unordered_map<ModalityType, Vector>& modal_embeddings,
                          const FusionParams& params);
    
    // 批量融合
    std::vector<Vector> batch_fuse(
        const std::vector<std::unordered_map<ModalityType, Vector>>& batch_embeddings,
        const FusionParams& params);
    
    // 获取支持的融合策略
    std::vector<FusionStrategy> get_supported_strategies() const;

private:
    std::unordered_map<FusionStrategy, std::shared_ptr<FusionStrategyInterface>> strategies_;
};

// 多模态数据库配置
struct MultimodalConfig {
    DatabaseConfig base_config;         // 基础数据库配置
    FusionParams default_fusion_params; // 默认融合参数
    bool enable_modality_indexing = true;  // 是否为每个模态建立独立索引
    bool store_raw_data = false;        // 是否存储原始数据
    uint32_t max_modalities_per_item = 5;  // 每个数据项最大模态数
};

// 多模态查询参数
struct MultimodalSearchParams : public SearchParams {
    std::vector<ModalityType> target_modalities;  // 目标查询模态
    bool use_cross_modal_search = false;          // 是否使用跨模态搜索
    FusionParams query_fusion_params;             // 查询时的融合参数
    
    MultimodalSearchParams() = default;
    MultimodalSearchParams(uint32_t k_) : SearchParams(k_) {}
};

// 异常类
class MultimodalException : public SageVDBException {
public:
    explicit MultimodalException(const std::string& message) 
        : SageVDBException("Multimodal: " + message) {}
};

} // namespace sage_vdb
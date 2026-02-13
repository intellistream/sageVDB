# SageVDB 多模态数据融合模块

## 概述

这是一个高度模块化的多模态数据融合算法模块，设计用于SAGE数据库系统。该模块支持多种模态的数据存储、融合和检索，并提供了插件式的架构，让您可以轻松添加新的模态处理器和融合算法。

## 主要特性

### 🔧 模块化设计
- **插件式架构**: 支持动态注册模态处理器和融合策略
- **解耦设计**: 各组件独立，易于测试和维护
- **可扩展性**: 轻松添加新的模态类型和融合算法

### 🎯 支持的模态类型
- 文本 (TEXT)
- 图像 (IMAGE)  
- 音频 (AUDIO)
- 视频 (VIDEO)
- 表格数据 (TABULAR)
- 时间序列 (TIME_SERIES)
- 自定义模态 (CUSTOM)

### 🚀 融合策略
- **向量拼接** (Concatenation): 简单直接的特征组合
- **加权平均** (Weighted Average): 基于权重的模态融合
- **注意力机制** (Attention-based): 自适应学习模态重要性
- **张量融合** (Tensor Fusion): 捕获高阶模态交互
- **双线性池化** (Bilinear Pooling): 高效的特征融合
- **自定义策略**: 支持用户定义的融合算法

## 快速开始

### 构建

```bash
# 推荐：通过 CLI 安装原生扩展
sage extensions install sage_vdb  # 需要重新编译时可追加 --force

# 可选：在组件目录手动运行构建脚本（调试/定制场景）
cd packages/sage-middleware/src/sage/middleware/components/sage_vdb
./build_multimodal.sh
```

### 基本使用

```cpp
#include "sage_vdb/multimodal_sage_vdb.h"

// 1. 创建数据库
DatabaseConfig config;
config.dimension = 512;
config.index_type = IndexType::FLAT;

auto db = MultimodalSageVDBFactory::create_text_image_db(config);

// 2. 准备多模态数据
Vector text_embedding = extract_text_features("Hello World");
Vector image_embedding = extract_image_features("image.jpg");

std::unordered_map<ModalityType, ModalData> modalities;
modalities[ModalityType::TEXT] = ModalData(ModalityType::TEXT, text_embedding);
modalities[ModalityType::IMAGE] = ModalData(ModalityType::IMAGE, image_embedding);

// 3. 添加数据
VectorId id = db->add_multimodal(modalities);

// 4. 搜索
std::unordered_map<ModalityType, ModalData> query;
query[ModalityType::TEXT] = ModalData(ModalityType::TEXT, query_embedding);

MultimodalSearchParams params(10);
auto results = db->search_multimodal(query, params);
```

### 自定义融合策略

```cpp
class MyCustomFusion : public FusionStrategyInterface {
public:
    Vector fuse(const std::unordered_map<ModalityType, Vector>& modal_embeddings,
               const FusionParams& params) override {
        // 实现您的融合逻辑
        return fused_vector;
    }
    
    FusionStrategy get_strategy_type() const override {
        return FusionStrategy::CUSTOM;
    }
};

// 注册自定义策略
db->register_fusion_strategy(FusionStrategy::CUSTOM, 
                            std::make_shared<MyCustomFusion>());
```

### 自定义模态处理器

```cpp
class MyModalityProcessor : public ModalityProcessor {
public:
    Vector process(const std::vector<uint8_t>& raw_data) override {
        // 处理原始数据为嵌入向量
        return embedding;
    }
    
    bool validate(const std::vector<uint8_t>& raw_data) const override {
        // 验证数据格式
        return true;
    }
    
    ModalityType get_type() const override {
        return ModalityType::CUSTOM;
    }
};

// 注册处理器
db->register_modality_processor(ModalityType::CUSTOM,
                               std::make_shared<MyModalityProcessor>());
```

## 运行测试

```bash
cd build_multimodal

# 运行单元测试
./test_multimodal

# 运行示例
export LD_LIBRARY_PATH=$PWD:$LD_LIBRARY_PATH
./multimodal_example
```

## 持久化与部署注意事项

- `build.sh` 会在 `python/` 与 `install/` 目录下安装 `_sage_vdb*.so`，并生成 `*.ids` 与新的 `*.order` 映射文件，确保 FAISS 向量与自定义向量 ID 在重新加载后保持一致。
- 如果需要完全清理构建产物，请同时删除 `build/`、`install/`、`python/_sage_vdb*.so` 以及对应的 `*.ids`、`*.order` 文件，避免旧的映射干扰新的部署。
- 在 CI 环境中，如果只复制安装目录，务必包含 `*.order` 文件，否则 Python 侧的相似度查询会因为 ID 映射缺失而得到空结果。

## 架构设计

### 核心组件

```
MultimodalSageVDB
├── ModalityManager          # 模态管理器
│   └── ModalityProcessor[]  # 模态处理器集合
├── FusionEngine            # 融合引擎  
│   └── FusionStrategy[]    # 融合策略集合
└── MetadataStore          # 元数据存储
```

### 数据流

```
原始数据 → ModalityProcessor → 嵌入向量 → FusionEngine → 融合向量 → 向量数据库
```

### 插件机制

所有模态处理器和融合策略都通过接口定义，支持运行时注册：

```cpp
// 模态处理器接口
class ModalityProcessor {
    virtual Vector process(const std::vector<uint8_t>& raw_data) = 0;
    virtual bool validate(const std::vector<uint8_t>& raw_data) const = 0;
    virtual ModalityType get_type() const = 0;
};

// 融合策略接口  
class FusionStrategyInterface {
    virtual Vector fuse(const std::unordered_map<ModalityType, Vector>& modal_embeddings,
                       const FusionParams& params) = 0;
    virtual FusionStrategy get_strategy_type() const = 0;
};
```

## 配置选项

### 多模态配置

```cpp
struct MultimodalConfig {
    DatabaseConfig base_config;              // 基础数据库配置
    FusionParams default_fusion_params;      // 默认融合参数
    bool enable_modality_indexing = true;    // 是否为每个模态建立独立索引
    bool store_raw_data = false;             // 是否存储原始数据
    uint32_t max_modalities_per_item = 5;    // 每个数据项最大模态数
};
```

### 融合参数

```cpp
struct FusionParams {
    FusionStrategy strategy = FusionStrategy::WEIGHTED_AVERAGE;
    std::unordered_map<ModalityType, float> modality_weights;  // 模态权重
    uint32_t target_dimension = 0;                            // 目标维度
    std::map<std::string, float> custom_params;               // 自定义参数
};
```

## 性能优化

### 批量操作
```cpp
// 批量添加
std::vector<MultimodalData> batch_data;
auto ids = db->add_multimodal_batch(batch_data);

// 批量融合
auto fused_vectors = fusion_engine->batch_fuse(batch_embeddings, params);
```

### 维度对齐
```cpp
// 统一不同模态的嵌入维度
FusionParams params;
params.target_dimension = 512;  // 目标维度
```

### 内存管理
- 使用智能指针管理对象生命周期
- 支持向量的移动语义
- 提供维度对齐工具减少内存开销

## 错误处理

模块提供了专门的异常类：

```cpp
try {
    auto db = MultimodalSageVDBFactory::create_text_image_db(config);
    VectorId id = db->add_multimodal(data);
} catch (const MultimodalException& e) {
    std::cerr << "多模态错误: " << e.what() << std::endl;
} catch (const SageVDBException& e) {
    std::cerr << "数据库错误: " << e.what() << std::endl;
}
```

## 扩展指南

### 添加新模态类型

1. 在 `ModalityType` 枚举中添加新类型
2. 实现 `ModalityProcessor` 接口
3. 注册处理器到数据库

### 添加新融合策略

1. 在 `FusionStrategy` 枚举中添加新策略
2. 实现 `FusionStrategyInterface` 接口  
3. 在工厂类中添加创建方法
4. 注册策略到融合引擎

### 性能调优

- 为高频模态优化处理器实现
- 使用适合的融合策略（注意力机制vs简单拼接）
- 合理设置目标维度平衡精度和性能
- 考虑使用模态特定的索引优化检索

## 依赖项

- C++17 或更高版本
- FAISS (向量检索库)
- OpenMP (可选，并行计算)
- OpenCV (可选，图像处理)
- FFmpeg (可选，音视频处理)

## 许可证

本项目遵循与SAGE相同的许可证。

## 贡献指南

欢迎贡献新的模态处理器和融合策略！请确保：

1. 遵循现有的代码风格
2. 添加相应的单元测试
3. 更新文档说明
4. 提供使用示例

---

*这个模块为SAGE数据库系统提供了强大的多模态数据处理能力，支持现代AI应用的复杂数据需求。*
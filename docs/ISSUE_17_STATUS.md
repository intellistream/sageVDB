# Issue #17: 多模态处理器实现与依赖关系说明

## 当前状态

### 多模态处理器实现 ✅

所有6种模态处理器已完全实现：

1. **文本处理器** (`TextModalityProcessor`) - ✅ 完成
   - 支持BERT风格的tokenization
   - 词嵌入平均化
   - 可配置序列长度和embedding维度

2. **图像处理器** (`ImageModalityProcessor`) - ✅ 完成
   - 支持多种格式：JPG, PNG, BMP, TIFF
   - 特征提取：直方图特征、纹理特征
   - 可选OpenCV支持（`OPENCV_ENABLED`宏控制）

3. **音频处理器** (`AudioModalityProcessor`) - ✅ 完成
   - MFCC特征提取
   - 频谱特征
   - 时域特征
   - 支持WAV, MP3, FLAC, M4A格式

4. **视频处理器** (`VideoModalityProcessor`) - ✅ 完成
   - 帧采样和特征聚合
   - 运动特征提取
   - 可选音频流提取
   - 支持MP4, AVI, MOV, MKV格式

5. **表格数据处理器** (`TabularModalityProcessor`) - ✅ 完成
   - CSV/TSV/JSON格式支持
   - 自动列类型检测（数值/分类/文本）
   - 缺失值处理
   - 特征归一化

6. **时间序列处理器** (`TimeSeriesModalityProcessor`) - ✅ 完成
   - 统计特征提取
   - 频域特征
   - 趋势和季节性分解
   - 滑动窗口特征

**代码位置：**
- 头文件：[include/sage_db/modality_processors.h](../include/sage_db/modality_processors.h) (240行)
- 实现文件：[src/modality_processors.cpp](../src/modality_processors.cpp) (941行)
- 工厂模式：`ModalityProcessorFactory` 支持标准和自定义处理器注册

### 依赖关系说明 ✅

**不存在循环依赖问题！**

依赖关系是单向的：

```
SAGE (isage)
    └── sage-middleware
            └── isage-vdb (sageVDB)  # 单向依赖
                    └── numpy  # 仅依赖numpy，不依赖SAGE
```

**详细说明：**

1. **sageVDB依赖** (从`pyproject.toml`):
   ```toml
   dependencies = [
       "numpy>=1.19.0",
   ]
   ```
   - ✅ 只依赖numpy
   - ✅ 不依赖任何SAGE包
   - ✅ 可以独立安装和使用

2. **SAGE依赖** (从`sage-middleware/pyproject.toml`):
   ```toml
   dependencies = [
       "isage-vdb>=0.1.5",  # SageVDB vector database
       ...
   ]
   ```
   - ✅ SAGE的middleware组件依赖sageVDB
   - ✅ 这是正常的上层→底层依赖，不是循环依赖

3. **为什么看起来像循环依赖？**
   - 在开发环境中，两者都安装在同一个conda环境(`sage`)
   - `pip list`显示所有包，造成"互相依赖"的假象
   - 实际上依赖方向明确：SAGE → sageVDB

## 构建和测试

### 多模态处理器测试

```bash
cd build
./test_multimodal
```

### 依赖验证

```bash
# 验证sageVDB可以独立安装
pip install isage-vdb  # 只会安装numpy依赖

# 验证SAGE会自动安装sageVDB
pip install isage-middleware  # 会自动安装isage-vdb
```

## 可选依赖

多模态处理器的某些功能需要可选依赖：

### OpenCV (用于图像/视频)
```bash
# 编译时启用
cmake -DENABLE_OPENCV=ON ...

# 运行时需要
pip install opencv-python
```

### 音频处理库
对于完整的音频支持，建议安装：
```bash
pip install librosa soundfile
```

## 下一步改进建议

虽然多模态处理器已实现，但可以进一步增强：

### 1. 深度学习模型集成
- [ ] 集成预训练的embedding模型（CLIP, BERT等）
- [ ] 支持PyTorch/ONNX模型推理
- [ ] GPU加速的特征提取

### 2. Python绑定
- [ ] 暴露多模态处理器到Python API
- [ ] 添加Python使用示例
- [ ] 添加Python测试用例

### 3. 文档完善
- [ ] 添加每个处理器的详细使用示例
- [ ] 性能基准测试文档
- [ ] 最佳实践指南

### 4. 性能优化
- [ ] 批量处理优化
- [ ] 并行特征提取
- [ ] 内存使用优化

## 示例代码

### C++ 使用示例

```cpp
#include <sage_db/modality_processors.h>
#include <sage_db/multimodal_sage_db.h>

using namespace sage_db;

// 创建文本处理器
TextModalityProcessor::TextConfig text_config;
text_config.embedding_dim = 768;
auto text_proc = ModalityProcessorFactory::create_text_processor(text_config);

// 处理文本数据
std::string text = "Hello, world!";
std::vector<uint8_t> text_bytes(text.begin(), text.end());
Vector text_embedding = text_proc->process(text_bytes);

// 创建多模态数据库
MultimodalDatabaseConfig config;
config.base_config.dimension = 768;
config.fusion_strategy = FusionStrategyType::WEIGHTED;

MultimodalSageDB mmdb(config);
mmdb.add_modality_processor(ModalityType::TEXT, text_proc);

// 添加多模态数据
ModalityInputMap inputs;
inputs[ModalityType::TEXT] = text_bytes;
auto id = mmdb.add_multimodal(inputs);
```

### 未来的Python API (建议)

```python
from sagevdb.multimodal import (
    MultimodalSageDB,
    TextProcessor,
    ImageProcessor,
    FusionStrategy
)

# 创建处理器
text_proc = TextProcessor(embedding_dim=768)
image_proc = ImageProcessor(target_size=(224, 224))

# 创建多模态数据库
db = MultimodalSageDB(
    dimension=1024,
    fusion_strategy=FusionStrategy.ATTENTION
)
db.add_processor("text", text_proc)
db.add_processor("image", image_proc)

# 添加数据
with open("image.jpg", "rb") as f:
    image_data = f.read()

vec_id = db.add({
    "text": "A beautiful sunset",
    "image": image_data
})

# 搜索
results = db.search({
    "text": "sunset beach"
}, k=10)
```

## 总结

- ✅ **多模态处理器**：6种类型全部实现完毕
- ✅ **依赖关系**：单向依赖，无循环依赖问题
- ✅ **代码质量**：完整的头文件和实现，约1200行代码
- 📝 **待改进**：Python绑定、深度学习集成、文档完善

## 参考文档

- [多模态功能说明](guides/README_Multimodal.md)
- [使用模式文档](USAGE_MODES.md)
- [代码实现](../src/modality_processors.cpp)

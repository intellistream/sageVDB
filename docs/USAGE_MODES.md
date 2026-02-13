# SageVDB 使用模式指南（Standalone / BYO-Embedding / Plugin / Service）

本指南阐述 SageVDB（下称 SageVDB）的四种典型使用方式，帮助你在不同场景中做出合适选择，并快速落地：
- Standalone（独立）：SageVDB-core + 可选内置处理器（builtin）
- BYO-Embedding（仅向量）：只用 SageVDB-core，外部自行生成 embeddings
- Plugin（插件注入）：SageVDB-core + 运行期加载的第三方/上层项目插件（如 SAGE 提供）
- Service（微服务）：把 SageVDB 以 REST/gRPC 的形式对外提供

> 术语约定：
> - core：SageVDB 核心库（索引/检索/融合/元数据 + 抽象接口，如 ModalityProcessor）。
> - builtin：轻量内置处理器（可选构建，用于演示/本地/CI），非必须。
> - plugin：外部实现的处理器/适配器（动态库），运行时注入。

---

## 快速选型建议

- 想要最小依赖、最稳定的第三方库形态：选择 BYO-Embedding（默认推荐）。
- 想开箱即用本地多模态（不依赖外部模型）：选择 Standalone（启用 builtin）。
- 想与上层项目解耦并可热插拔：选择 Plugin（SAGE 提供插件，运行期注入）。
- 想跨语言/跨团队统一访问、易于扩展：选择 Service（REST/gRPC）。

---

## 公共概念与关键 API（适用于所有模式）

- `sage_vdb::MultimodalSageVDB`：多模态 DB 入口，负责融合、索引、检索、元数据等。
- `sage_vdb::ModalityProcessor`：模态处理器抽象（把原始 bytes 处理成 embedding 向量）。
- `sage_vdb::ModalityManager`：处理器注册与调度（`register_modality_processor`）。
- `sage_vdb::FusionStrategy` 与 `FusionEngine`：多模态融合策略与执行（拼接、加权、注意力…）。
- `sage_vdb::ModalData`：承载某一模态的输入（可直接是 embedding 或原始数据）。

> 重要：当前实现中，若你传入的 `ModalData` 已含 `embedding`，SageVDB 会直接使用该向量；只有当你选择“处理原始数据”这条路径并注册了对应处理器时，才会调用 `ModalityProcessor` 去生成 embedding。

---

## 1) Standalone（独立 + builtin）

- 组成：`SageVDB-core` + `SageVDB-builtin`（可选构建）。
- 适用：希望在没有上层模型依赖的情况下，快速完成“原始数据 -> 向量 -> 检索”，用于本地 demo、POC、教学或 CI。
- 依赖：尽量使用轻量依赖（例如 OpenCV 等），避免引入大型深度学习框架。

数据流（简化）：
```
raw bytes -> [builtin 处理器] -> embedding -> [fusion] -> index/search
```

示例（需要已启用/提供 builtin 实现）：
```cpp
#include <memory>
#include "sage_vdb/multimodal_sage_vdb.h"
#include "sage_vdb/modality_processors.h" // 声明在 include 中；具体实现由 builtin 或外部提供

using namespace sage_vdb;

int main() {
    DatabaseConfig base; base.dimension = 256; base.index_type = IndexType::FLAT;
    MultimodalConfig cfg; cfg.base_config = base;
    cfg.default_fusion_params.strategy = FusionStrategy::WEIGHTED_AVERAGE;

    MultimodalSageVDB db(cfg);

    // 假设已提供 Text/Image builtin 实现（示例）
    auto text = std::make_shared<TextModalityProcessor>(TextModalityProcessor::TextConfig{});
    auto image = std::make_shared<ImageModalityProcessor>(ImageModalityProcessor::ImageConfig{});
    db.register_modality_processor(ModalityType::TEXT, text);
    db.register_modality_processor(ModalityType::IMAGE, image);

    // 使用原始 bytes 处理（伪示例）：
    // std::vector<uint8_t> text_bytes = ...; std::vector<uint8_t> image_bytes = ...;
    // 由调用方封装为 ModalData，并让处理器生成 embedding（参考具体 API 实现）。
}
```

注意事项：
- builtin 应作为可选组件构建（例如 CMake: `BUILD_SageVDB_BUILTIN`）。
- builtin 面向演示/轻量使用，生产推荐 BYO 或 Plugin。

---

## 2) BYO-Embedding（仅向量，默认推荐）

- 组成：仅 `SageVDB-core`。
- 适用：把 embedding 生成留给上层项目（如 SAGE），SageVDB 专注索引/检索/融合/元数据。
- 优点：无循环依赖、最小耦合、最稳定；SageVDB 可作为独立第三方库发布。

数据流（简化）：
```
embedding (来自外部) -> [fusion] -> index/search
```

示例（直接使用已生成的 embedding）：
```cpp
#include <numeric>
#include "sage_vdb/multimodal_sage_vdb.h"

using namespace sage_vdb;

int main() {
    DatabaseConfig base; base.dimension = 128; base.index_type = IndexType::FLAT;
    MultimodalConfig cfg; cfg.base_config = base;
    cfg.default_fusion_params.strategy = FusionStrategy::CONCATENATION;
    cfg.default_fusion_params.target_dimension = base.dimension;

    MultimodalSageVDB db(cfg);

    // 准备外部已生成的向量
    Vector text(100); std::iota(text.begin(), text.end(), 1.0f);
    Vector image(200); std::iota(image.begin(), image.end(), 2.0f);

    std::unordered_map<ModalityType, ModalData> modalities;
    modalities[ModalityType::TEXT]  = ModalData(ModalityType::TEXT,  text);
    modalities[ModalityType::IMAGE] = ModalData(ModalityType::IMAGE, image);

    VectorId id = db.add_multimodal(modalities);
}
```

注意事项：
- 若传入 `ModalData.embedding` 为空，SageVDB 当前不会自动生成 embedding；需自行注册处理器或走 Plugin/Standalone。

---

## 3) Plugin（插件注入）

- 组成：`SageVDB-core` + 插件动态库（由上层项目或第三方提供，如 SAGE）。
- 适用：需要严格解耦、可热插拔、不同部署加载不同实现的场景。
- 原理：插件在运行时被加载，并通过一个约定的 C 入口将处理器注册到 DB。

示意：
```
[app]
  ├─ loads SageVDB-core
  ├─ dlopen("libsage_modality.so")
  └─ plugin_entry(db*) -> db->register_modality_processor(...)
```

插件入口（示例，ABI 示意，实际名称与签名以仓库文档/实现为准）：
```cpp
extern "C" void register_SageVDB_plugin(sage_vdb::MultimodalSageVDB* db) {
    using namespace sage_vdb;
    auto text = std::make_shared<TextModalityProcessor>(TextModalityProcessor::TextConfig{});
    db->register_modality_processor(ModalityType::TEXT, text);
}
```

注意事项：
- 建议定义 `api_version`/`plugin_version` 以做兼容检测。
- 插件应只依赖 public headers；不要反向依赖宿主私有实现。
- 加载器（dlopen/dlsym 或 LoadLibrary/GetProcAddress）需做好异常与资源回收。

---

## 4) Service（微服务）

- 组成：将 `SageVDB-core` 打包为独立进程服务（REST/gRPC）。
- 适用：跨语言/跨团队共享、水平扩展、统一鉴权与监控的场景。
- 两种典型 API：
  - BYO-Embedding API：客户端直接发送 embedding；服务做融合/检索/返回结果。
  - Raw Data API（可选）：服务端已装载 builtin 或插件处理器，接收原始数据并在服务端生成 embedding。

建议能力：批量/异步、超时/限流、健康检查、TLS/认证、指标（Prometheus）与日志。

---

## 版本与兼容性建议

- 语义化版本（semver）：public headers 的破坏性变更应 bump major。
- 插件 ABI：在 loader 与插件入口之间传递 `api_version`；不兼容时拒绝加载并给出明确错误信息。
- 文档化稳定/实验性接口：对 `ModalityProcessor`、`ModalityManager`、`FusionStrategy` 等接口标注稳定性等级。

---

## 常见问题（FAQ）

- Q：独立使用 SageVDB 时能否不依赖任何模型？
  - A：可以，选择 BYO-Embedding（只传入 embedding），或启用内置（builtin）做本地演示。

- Q：如何避免与上层项目（如 SAGE）产生循环依赖？
  - A：通过 BYO-Embedding 或 Plugin。SageVDB 只提供接口，SAGE 在运行时注入实现（注册处理器）。

- Q：融合后的维度如何保证与底层索引维度一致？
  - A：请设置 `FusionParams.target_dimension = DatabaseConfig.dimension`；或选择拼接等能匹配维度的策略并在入库前做对齐。

---

## 我应当如何开始？

- 只用核心库：从 BYO-Embedding 开始（最简单、最稳）。
- 需要演示/离线体验：启用 builtin 并运行 Standalone 示例。
- 与 SAGE 集成且希望完全解耦：采用 Plugin；SAGE 编译生成插件并在运行时注入。
- 多团队/跨语言/需要横向扩展：将 SageVDB 以 Service 形态部署。

> 若你需要示例代码、CMake 片段或插件模板，请查看仓库的示例与 README（后续会在 `examples/` 与文档中提供更多模板）。

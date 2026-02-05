#include "sage_vdb/multimodal_fusion.h"
#include <stdexcept>

namespace sage_vdb {

// ========== ModalityManager 实现 ==========
void ModalityManager::register_processor(ModalityType type, 
                                        std::shared_ptr<ModalityProcessor> processor) {
    if (!processor) {
        throw std::invalid_argument("Processor cannot be null");
    }
    
    processors_[type] = processor;
}

Vector ModalityManager::process_modality(ModalityType type, 
                                        const std::vector<uint8_t>& raw_data) {
    auto it = processors_.find(type);
    if (it == processors_.end()) {
        throw std::runtime_error("No processor registered for modality type: " + 
                                std::to_string(static_cast<int>(type)));
    }
    
    return it->second->process(raw_data);
}

bool ModalityManager::validate_modality(ModalityType type, 
                                       const std::vector<uint8_t>& raw_data) const {
    auto it = processors_.find(type);
    if (it == processors_.end()) {
        return false;
    }
    
    return it->second->validate(raw_data);
}

std::vector<ModalityType> ModalityManager::get_supported_modalities() const {
    std::vector<ModalityType> modalities;
    for (const auto& [type, processor] : processors_) {
        modalities.push_back(type);
    }
    return modalities;
}

// ========== FusionEngine 实现 ==========
void FusionEngine::register_strategy(FusionStrategy strategy, 
                                    std::shared_ptr<FusionStrategyInterface> impl) {
    if (!impl) {
        throw std::invalid_argument("Strategy implementation cannot be null");
    }
    
    strategies_[strategy] = impl;
}

Vector FusionEngine::fuse_embeddings(const std::unordered_map<ModalityType, Vector>& modal_embeddings,
                                    const FusionParams& params) {
    auto it = strategies_.find(params.strategy);
    if (it == strategies_.end()) {
        throw std::runtime_error("No strategy registered for fusion type: " + 
                                std::to_string(static_cast<int>(params.strategy)));
    }
    
    return it->second->fuse(modal_embeddings, params);
}

std::vector<Vector> FusionEngine::batch_fuse(
    const std::vector<std::unordered_map<ModalityType, Vector>>& batch_embeddings,
    const FusionParams& params) {
    
    std::vector<Vector> results;
    results.reserve(batch_embeddings.size());
    
    for (const auto& modal_embeddings : batch_embeddings) {
        results.push_back(fuse_embeddings(modal_embeddings, params));
    }
    
    return results;
}

std::vector<FusionStrategy> FusionEngine::get_supported_strategies() const {
    std::vector<FusionStrategy> strategies;
    for (const auto& [strategy, impl] : strategies_) {
        strategies.push_back(strategy);
    }
    return strategies;
}

} // namespace sage_vdb
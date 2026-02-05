#include "sage_vdb/anns/anns_interface.h"
#include <stdexcept>

namespace sage_vdb {
namespace anns {

// ANNSRegistry implementation
ANNSRegistry& ANNSRegistry::instance() {
    static ANNSRegistry instance;
    return instance;
}

void ANNSRegistry::register_factory(std::unique_ptr<ANNSFactory> factory) {
    const std::string name = factory->algorithm_name();
    register_factory(name, std::move(factory));
}

void ANNSRegistry::register_factory(const std::string& name,
                                    std::unique_ptr<ANNSFactory> factory) {
    if (factories_.find(name) != factories_.end()) {
        throw std::runtime_error("Algorithm '" + name + "' is already registered");
    }
    factories_[name] = std::move(factory);
}

void ANNSRegistry::unregister_factory(const std::string& name) {
    auto it = factories_.find(name);
    if (it != factories_.end()) {
        factories_.erase(it);
    }
}

std::unique_ptr<ANNSAlgorithm> ANNSRegistry::create_algorithm(const std::string& name) const {
    auto it = factories_.find(name);
    if (it == factories_.end()) {
        throw std::runtime_error("Algorithm '" + name + "' is not registered");
    }
    return it->second->create();
}

std::vector<std::string> ANNSRegistry::list_algorithms() const {
    std::vector<std::string> algorithms;
    algorithms.reserve(factories_.size());
    
    for (const auto& [name, factory] : factories_) {
        algorithms.push_back(name);
    }
    
    return algorithms;
}

bool ANNSRegistry::is_available(const std::string& name) const {
    return factories_.find(name) != factories_.end();
}

const ANNSFactory* ANNSRegistry::get_factory(const std::string& name) const {
    auto it = factories_.find(name);
    if (it == factories_.end()) {
        return nullptr;
    }
    return it->second.get();
}

std::vector<std::string> ANNSRegistry::algorithms_supporting_distance(DistanceMetric metric) const {
    std::vector<std::string> algorithms;
    
    for (const auto& [name, factory] : factories_) {
        const auto supported = factory->supported_distances();
        if (std::find(supported.begin(), supported.end(), metric) != supported.end()) {
            algorithms.push_back(name);
        }
    }
    
    return algorithms;
}

std::vector<std::string> ANNSRegistry::algorithms_supporting_updates() const {
    std::vector<std::string> algorithms;
    
    for (const auto& [name, factory] : factories_) {
        auto instance = factory->create();
        if (instance->supports_updates()) {
            algorithms.push_back(name);
        }
    }
    
    return algorithms;
}

std::vector<std::string> ANNSRegistry::algorithms_supporting_deletions() const {
    std::vector<std::string> algorithms;
    
    for (const auto& [name, factory] : factories_) {
        auto instance = factory->create();
        if (instance->supports_deletions()) {
            algorithms.push_back(name);
        }
    }
    
    return algorithms;
}

} // namespace anns
} // namespace sage_vdb
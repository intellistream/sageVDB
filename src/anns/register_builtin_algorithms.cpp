#include "sage_vdb/anns/anns_interface.h"
#include "brute_force_plugin.h"
#ifdef ENABLE_FAISS
#include "faiss_plugin.h"
#endif

#include <mutex>
#include <stdexcept>

namespace sage_vdb::anns {

void register_builtin_algorithms() {
    static std::once_flag registered_once;
    std::call_once(registered_once, []() {
        auto& registry = ANNSRegistry::instance();
        if (!registry.is_available("brute_force")) {
            throw std::runtime_error(
                "Built-in ANNS algorithm 'brute_force' is not registered"
            );
        }
#ifdef ENABLE_FAISS
        if (!registry.is_available("FAISS")) {
            throw std::runtime_error(
                "Built-in ANNS algorithm 'FAISS' is not registered"
            );
        }
#endif
    });
}

} // namespace sage_vdb::anns

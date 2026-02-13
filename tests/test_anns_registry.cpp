#include "sage_vdb/anns/anns_interface.h"
#include <algorithm>
#include <cassert>
#include <iostream>

using namespace sage_vdb;
using namespace sage_vdb::anns;

int main() {
    std::cout << "Testing ANNS registry..." << std::endl;

    auto& registry = ANNSRegistry::instance();
    auto algorithms = registry.list_algorithms();

    const bool has_bruteforce = std::find(algorithms.begin(), algorithms.end(), "brute_force") != algorithms.end();
    assert(has_bruteforce);

    auto brute = registry.create_algorithm("brute_force");
    assert(brute);

    auto supported = brute->supported_distances();
    const bool supports_l2 = std::find(supported.begin(), supported.end(), DistanceMetric::L2) != supported.end();
    assert(supports_l2);

#ifdef ENABLE_FAISS
    const bool has_faiss = std::find(algorithms.begin(), algorithms.end(), "FAISS") != algorithms.end();
    assert(has_faiss);
    auto faiss = registry.create_algorithm("FAISS");
    assert(faiss);
#endif

    std::cout << "✅ ANNS registry test passed" << std::endl;
    return 0;
}

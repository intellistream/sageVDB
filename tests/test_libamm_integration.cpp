#include "sage_vdb/sage_vdb.h"
#include "sage_vdb/anns/flat_gpu_plugin.h"
#include <iostream>
#include <random>
#include <cassert>
#include <cmath>

using namespace sage_vdb;

// Test that LibAMM CRS/SMP-PCA algorithms are available and functional
void test_libamm_flatgpu_plugin() {
    std::cout << "Testing LibAMM integration with FlatGPU plugin..." << std::endl;
    
    const int dim = 128;
    const int num_vectors = 1000;
    const int num_queries = 10;
    const int k = 5;
    
    // Create random test data
    std::vector<float> database(num_vectors * dim);
    std::vector<float> queries(num_queries * dim);
    
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    
    for (auto& val : database) val = dist(rng);
    for (auto& val : queries) val = dist(rng);
    
    // Test FlatGPU plugin with LibAMM CRS
    {
        std::cout << "Testing CRS algorithm..." << std::endl;
        FlatGPUConfig config;
        config.dim = dim;
        config.metric = DistanceMetric::L2;
        config.use_libamm = true;
        config.libamm_algo = "CRS";
        config.sketch_size = 32;  // Smaller than dimension for approximation
        config.use_cuda = false;   // CPU-only for CI
        
        FlatGPUPlugin plugin(config);
        plugin.add(database.data(), num_vectors);
        
        std::vector<int64_t> labels(num_queries * k);
        std::vector<float> distances(num_queries * k);
        
        plugin.search(queries.data(), num_queries, k, labels.data(), distances.data());
        
        // Verify we got valid results
        bool has_valid_results = false;
        for (int i = 0; i < num_queries * k; i++) {
            if (labels[i] >= 0 && labels[i] < num_vectors) {
                has_valid_results = true;
                break;
            }
        }
        
        assert(has_valid_results && "CRS should return valid search results");
        std::cout << "✅ CRS algorithm test passed" << std::endl;
    }
    
    // Test FlatGPU plugin with LibAMM SMP-PCA
    {
        std::cout << "Testing SMP-PCA algorithm..." << std::endl;
        FlatGPUConfig config;
        config.dim = dim;
        config.metric = DistanceMetric::L2;
        config.use_libamm = true;
        config.libamm_algo = "SMP-PCA";
        config.sketch_size = 32;
        config.use_cuda = false;
        
        FlatGPUPlugin plugin(config);
        plugin.add(database.data(), num_vectors);
        
        std::vector<int64_t> labels(num_queries * k);
        std::vector<float> distances(num_queries * k);
        
        plugin.search(queries.data(), num_queries, k, labels.data(), distances.data());
        
        // Verify we got valid results
        bool has_valid_results = false;
        for (int i = 0; i < num_queries * k; i++) {
            if (labels[i] >= 0 && labels[i] < num_vectors) {
                has_valid_results = true;
                break;
            }
        }
        
        assert(has_valid_results && "SMP-PCA should return valid search results");
        std::cout << "✅ SMP-PCA algorithm test passed" << std::endl;
    }
    
    std::cout << "✅ All LibAMM integration tests passed" << std::endl;
}

// Test fallback to exact computation when LibAMM is disabled
void test_libamm_fallback() {
    std::cout << "Testing fallback when LibAMM is disabled..." << std::endl;
    
    const int dim = 64;
    const int num_vectors = 100;
    const int num_queries = 5;
    const int k = 3;
    
    std::vector<float> database(num_vectors * dim);
    std::vector<float> queries(num_queries * dim);
    
    std::mt19937 rng(123);
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    
    for (auto& val : database) val = dist(rng);
    for (auto& val : queries) val = dist(rng);
    
    FlatGPUConfig config;
    config.dim = dim;
    config.metric = DistanceMetric::L2;
    config.use_libamm = false;  // Disable LibAMM
    config.use_cuda = false;
    
    FlatGPUPlugin plugin(config);
    plugin.add(database.data(), num_vectors);
    
    std::vector<int64_t> labels(num_queries * k);
    std::vector<float> distances(num_queries * k);
    
    plugin.search(queries.data(), num_queries, k, labels.data(), distances.data());
    
    // Verify exact computation works
    for (int i = 0; i < num_queries * k; i++) {
        assert(labels[i] >= 0 && labels[i] < num_vectors);
        assert(distances[i] >= 0.0f);  // Distances should be non-negative
    }
    
    std::cout << "✅ Fallback to exact computation test passed" << std::endl;
}

int main() {
    try {
#ifdef ENABLE_LIBAMM
        std::cout << "LibAMM is ENABLED - running integration tests" << std::endl;
        test_libamm_flatgpu_plugin();
        test_libamm_fallback();
#else
        std::cout << "LibAMM is DISABLED - skipping LibAMM-specific tests" << std::endl;
        test_libamm_fallback();
#endif
        
        std::cout << "\n🎉 All tests passed successfully!" << std::endl;
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "❌ Test failed with exception: " << e.what() << std::endl;
        return 1;
    } catch (...) {
        std::cerr << "❌ Test failed with unknown exception" << std::endl;
        return 1;
    }
}

/*
 * SONG kernel configuration
 * Ported from sage-db_outdated without LibTorch dependencies
 */
#ifndef sage_vdb_ANNS_SONG_CONFIG_HPP
#define sage_vdb_ANNS_SONG_CONFIG_HPP

namespace song_kernel {

// Type definitions
typedef float data_value_t;
typedef float value_t;
typedef double dist_t;
typedef size_t idx_t;
typedef int UINT;

// GPU kernel configuration
constexpr int ACC_BATCH_SIZE = 2048;
constexpr int FIXED_DEGREE = 31;
constexpr int FIXED_DEGREE_SHIFT = 5;

// CPU construction parameters
constexpr int SEARCH_DEGREE = 15;
constexpr int CONSTRUCT_SEARCH_BUDGET = 150;

} // namespace song_kernel

#endif // sage_vdb_ANNS_SONG_CONFIG_HPP

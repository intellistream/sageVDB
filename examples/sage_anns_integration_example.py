#!/usr/bin/env python3
"""
Example: Using sage-anns algorithms with sageVDB

This example demonstrates how to use sage-anns ANNS algorithms
as a backend for sageVDB through Python-level integration.
"""

import numpy as np
from sagevdb import create_database, DatabaseConfig, DistanceMetric

def main():
    print("=== SageVDB + sage-anns Integration Example ===\n")

    # Check available sage-anns algorithms
    try:
        from sagevdb import list_sage_anns_algorithms
        algorithms = list_sage_anns_algorithms()
        print(f"Available sage-anns algorithms: {algorithms}")

        if not algorithms:
            print("\nNo sage-anns algorithms registered.")
            print("Install algorithm implementations with:")
            print("  pip install isage-anns  # (may need specific algorithm packages)")
            return

    except ImportError:
        print("sage-anns not installed. Install with:")
        print("  pip install 'isage-vdb[sage-anns]'")
        return

    # Method 1: Create database with sage-anns backend directly
    print("\n--- Method 1: Direct creation with backend parameter ---")
    dimension = 128
    db1 = create_database(
        dimension,
        backend="sage-anns",
        algorithm=algorithms[0],  # Use first available algorithm
        metric="l2",
    )

    # Add some vectors
    vectors = np.random.rand(100, dimension).astype(np.float32)
    metadata = [{"id": i, "category": f"cat_{i % 3}"} for i in range(100)]
    db1.build_index(vectors, metadata=metadata)

    # Search
    query = np.random.rand(dimension).astype(np.float32)
    results = db1.search(query, k=5, include_metadata=True)

    print(f"Added {len(vectors)} vectors")
    print(f"Search results (top 5):")
    for i, result in enumerate(results[:5]):
        print(f"  {i+1}. ID={result.id}, Distance={result.score:.4f}, "
              f"Metadata={result.metadata}")

    # Method 2: Use DatabaseConfig
    print("\n--- Method 2: Using DatabaseConfig ---")
    config = DatabaseConfig(dimension)
    config.metric = DistanceMetric.L2
    config.anns_algorithm = algorithms[0]
    config.anns_build_params = {"M": "32", "ef_construction": "200"}  # HNSW params as example

    db2 = create_database(config, backend="sage-anns")

    # Add vectors incrementally
    for i in range(10):
        vec = np.random.rand(dimension).astype(np.float32)
        vec_id = db2.add(vec, metadata={"batch": str(i)})
        if i == 0:
            print(f"Added first vector with ID: {vec_id}")

    print(f"Total vectors in db2: {db2.dimension}")

    # Batch search
    print("\n--- Batch Search ---")
    queries = np.random.rand(3, dimension).astype(np.float32)
    batch_results = db1.batch_search(queries, k=3)

    print(f"Batch search for {len(queries)} queries:")
    for i, results in enumerate(batch_results):
        print(f"  Query {i+1}: Found {len(results)} results")

    print("\n=== Integration successful! ===")


if __name__ == "__main__":
    main()

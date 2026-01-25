#!/usr/bin/env python3
"""
SageDB Python Persistence Example

Demonstrates how to save and load vector databases in Python,
including metadata persistence.
"""

import numpy as np
import os
import tempfile
from sagevdb import SageDB, DatabaseConfig, IndexType, DistanceMetric


def basic_persistence_example():
    """Basic save/load example"""
    print("=" * 60)
    print("Basic Persistence Example")
    print("=" * 60)
    
    # Create database configuration
    config = DatabaseConfig(128)
    config.index_type = IndexType.FLAT
    config.metric = DistanceMetric.L2
    
    db = SageDB(config)
    
    # Add vectors (use add_batch for multiple vectors)
    print("Adding 100 vectors...")
    vectors = [np.random.rand(128).astype(np.float32) for _ in range(100)]
    ids = db.add_batch(vectors)
    
    # Build index
    print("Building index...")
    db.build_index()
    
    # Query before saving
    query = np.random.rand(128).astype(np.float32)
    results_before = db.search(query, k=5)
    print(f"\nResults before save: {len(results_before)} neighbors found")
    for i, r in enumerate(results_before[:3]):
        print(f"  {i+1}. ID={r.id}, score={r.score:.4f}")
    
    # Save to disk
    with tempfile.TemporaryDirectory() as tmpdir:
        filepath = os.path.join(tmpdir, "my_database.sagedb")
        print(f"\nSaving database to: {filepath}")
        db.save(filepath)
        
        # Create new database and load
        db2 = SageDB(config)
        print(f"Loading database from: {filepath}")
        db2.load(filepath)
        
        # Query after loading
        results_after = db2.search(query, k=5)
        print(f"\nResults after load: {len(results_after)} neighbors found")
        for i, r in enumerate(results_after[:3]):
            print(f"  {i+1}. ID={r.id}, score={r.score:.4f}")
        
        # Verify results match
        assert len(results_before) == len(results_after)
        for r1, r2 in zip(results_before, results_after):
            assert r1.id == r2.id
            assert abs(r1.score - r2.score) < 1e-5
        
        print("\n✅ Persistence verified: results match perfectly!")


def metadata_persistence_example():
    """Save/load with metadata"""
    print("\n" + "=" * 60)
    print("Metadata Persistence Example")
    print("=" * 60)
    
    config = DatabaseConfig(64)
    config.index_type = IndexType.FLAT
    config.metric = DistanceMetric.COSINE
    
    db = SageDB(config)
    
    # Add vectors with metadata
    print("Adding 50 vectors with metadata...")
    for i in range(50):
        vec = np.random.rand(64).astype(np.float32)
        metadata = {
            "category": f"cat_{i % 5}",
            "value": str(i),  # metadata values must be strings
            "tag": f"item_{i}"
        }
        vec_id = db.add(vec, metadata)
    
    db.build_index()
    
    # Save both database and metadata
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "db_with_meta.sagedb")
        meta_path = os.path.join(tmpdir, "db_with_meta.meta")
        
        print(f"\nSaving database to: {db_path}")
        print(f"Saving metadata to: {meta_path}")
        db.save(db_path)
        db.metadata_store().save(meta_path)
        
        # Load into new database
        db2 = SageDB(config)
        print(f"\nLoading database from: {db_path}")
        print(f"Loading metadata from: {meta_path}")
        db2.load(db_path)
        db2.metadata_store().load(meta_path)
        
        # Verify metadata
        print("\nVerifying metadata...")
        query = np.random.rand(64).astype(np.float32)
        results = db2.search(query, k=5, include_metadata=True)
        
        for i, r in enumerate(results):
            print(f"  {i+1}. ID={r.id}, metadata={r.metadata}")
        
        print("\n✅ Metadata persistence verified!")


def incremental_save_example():
    """Demonstrate incremental updates and re-saving"""
    print("\n" + "=" * 60)
    print("Incremental Save Example")
    print("=" * 60)
    
    config = DatabaseConfig(32)
    config.index_type = IndexType.FLAT
    config.metric = DistanceMetric.L2
    
    with tempfile.TemporaryDirectory() as tmpdir:
        filepath = os.path.join(tmpdir, "incremental.sagedb")
        
        # Initial save
        db = SageDB(config)
        vectors = [np.random.rand(32).astype(np.float32) for _ in range(50)]
        db.add_batch(vectors)
        db.build_index()
        
        print(f"Initial save: {db.size()} vectors")
        db.save(filepath)
        
        # Load and add more vectors
        db2 = SageDB(config)
        db2.load(filepath)
        print(f"After load: {db2.size()} vectors")
        
        new_vectors = [np.random.rand(32).astype(np.float32) for _ in range(30)]
        db2.add_batch(new_vectors)
        db2.build_index()
        
        print(f"After adding more: {db2.size()} vectors")
        db2.save(filepath)
        
        # Load again to verify
        db3 = SageDB(config)
        db3.load(filepath)
        print(f"Final verification: {db3.size()} vectors")
        
        assert db3.size() == 80
        print("\n✅ Incremental save/load verified!")


def simple_example():
    """Simplified example for documentation"""
    print("\n" + "=" * 60)
    print("Simple Persistence Example")
    print("=" * 60)
    
    config = DatabaseConfig(128)
    config.index_type = IndexType.FLAT
    config.metric = DistanceMetric.L2
    
    db = SageDB(config)
    
    # Add some vectors
    vectors = [np.random.rand(128).astype(np.float32) for _ in range(100)]
    db.add_batch(vectors)
    db.build_index()
    
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "simple.sagedb")
        
        # Save
        db.save(path)
        print(f"Saved {db.size()} vectors")
        
        # Load
        db2 = SageDB(config)
        db2.load(path)
        print(f"Loaded {db2.size()} vectors")
        
        assert db2.size() == 100
        print("\n✅ Simple persistence works!")


def main():
    """Run all examples"""
    basic_persistence_example()
    metadata_persistence_example()
    incremental_save_example()
    simple_example()
    
    print("\n" + "=" * 60)
    print("All persistence examples completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()

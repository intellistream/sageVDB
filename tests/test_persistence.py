#!/usr/bin/env python3
"""
Test SageDB persistence functionality

This test verifies that save/load operations work correctly
for both vector data and metadata.
"""

import pytest
import numpy as np
import os
import tempfile
from sagevdb import (
    SageDB, 
    DatabaseConfig, 
    IndexType, 
    DistanceMetric, 
    SearchParams
)


class TestPersistence:
    """Test persistence functionality"""
    
    def test_basic_save_load(self):
        """Test basic save and load operations"""
        config = DatabaseConfig(
            dimension=128,
            index_type=IndexType.FLAT,
            metric=DistanceMetric.L2
        )
        
        # Create and populate database
        db1 = SageDB(config)
        vectors = np.random.rand(100, 128).astype(np.float32)
        ids = db1.add_vectors(vectors)
        db1.build_index()
        
        # Save to temporary file
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "test.sagedb")
            db1.save(filepath)
            
            # Load into new database
            db2 = SageDB(config)
            db2.load(filepath)
            
            # Verify size
            assert db2.size() == db1.size() == 100
            
            # Verify search results match
            query = np.random.rand(128).astype(np.float32)
            params = SearchParams(k=5)
            results1 = db1.search(query, params)
            results2 = db2.search(query, params)
            
            assert len(results1) == len(results2)
            for r1, r2 in zip(results1, results2):
                assert r1.id == r2.id
                assert abs(r1.distance - r2.distance) < 1e-5
    
    def test_metadata_persistence(self):
        """Test metadata save and load"""
        config = DatabaseConfig(
            dimension=64,
            index_type=IndexType.FLAT,
            metric=DistanceMetric.COSINE
        )
        
        db1 = SageDB(config)
        
        # Add vectors with metadata
        metadata_map = {}
        for i in range(50):
            vec = np.random.rand(64).astype(np.float32)
            vec_id = db1.add_vector(vec)
            meta = {"index": float(i), "category": f"cat_{i % 3}"}
            db1.set_metadata(vec_id, meta)
            metadata_map[vec_id] = meta
        
        db1.build_index()
        
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.sagedb")
            meta_path = os.path.join(tmpdir, "test.meta")
            
            db1.save(db_path)
            db1.save_metadata(meta_path)
            
            # Load into new database
            db2 = SageDB(config)
            db2.load(db_path)
            db2.load_metadata(meta_path)
            
            # Verify metadata
            for vec_id, expected_meta in metadata_map.items():
                actual_meta = db2.get_metadata(vec_id)
                assert actual_meta == expected_meta
    
    def test_empty_database_save_load(self):
        """Test save/load of empty database"""
        config = DatabaseConfig(
            dimension=32,
            index_type=IndexType.FLAT,
            metric=DistanceMetric.L2
        )
        
        db1 = SageDB(config)
        db1.build_index()  # Build empty index
        
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "empty.sagedb")
            db1.save(filepath)
            
            db2 = SageDB(config)
            db2.load(filepath)
            
            assert db2.size() == 0
            assert db2.is_trained()
    
    def test_incremental_updates(self):
        """Test save -> load -> add more -> save again"""
        config = DatabaseConfig(
            dimension=64,
            index_type=IndexType.FLAT,
            metric=DistanceMetric.L2
        )
        
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "incremental.sagedb")
            
            # Initial save
            db1 = SageDB(config)
            vecs1 = np.random.rand(50, 64).astype(np.float32)
            db1.add_vectors(vecs1)
            db1.build_index()
            db1.save(filepath)
            
            # Load and add more
            db2 = SageDB(config)
            db2.load(filepath)
            assert db2.size() == 50
            
            vecs2 = np.random.rand(30, 64).astype(np.float32)
            db2.add_vectors(vecs2)
            db2.build_index()
            db2.save(filepath)
            
            # Load final state
            db3 = SageDB(config)
            db3.load(filepath)
            assert db3.size() == 80
    
    def test_different_index_types(self):
        """Test persistence works with different index types"""
        index_types = [IndexType.FLAT, IndexType.HNSW]
        
        for idx_type in index_types:
            config = DatabaseConfig(
                dimension=32,
                index_type=idx_type,
                metric=DistanceMetric.L2
            )
            
            if idx_type == IndexType.HNSW:
                config.anns_build_params = {"M": "16", "ef_construction": "200"}
            
            db1 = SageDB(config)
            vectors = np.random.rand(100, 32).astype(np.float32)
            db1.add_vectors(vectors)
            db1.build_index()
            
            with tempfile.TemporaryDirectory() as tmpdir:
                filepath = os.path.join(tmpdir, f"test_{idx_type}.sagedb")
                db1.save(filepath)
                
                db2 = SageDB(config)
                db2.load(filepath)
                
                assert db2.size() == 100
                assert db2.is_trained()
    
    def test_cosine_metric_persistence(self):
        """Test persistence with cosine similarity metric"""
        config = DatabaseConfig(
            dimension=128,
            index_type=IndexType.FLAT,
            metric=DistanceMetric.COSINE
        )
        
        db1 = SageDB(config)
        vectors = np.random.rand(100, 128).astype(np.float32)
        db1.add_vectors(vectors)
        db1.build_index()
        
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "cosine.sagedb")
            db1.save(filepath)
            
            db2 = SageDB(config)
            db2.load(filepath)
            
            # Test search with cosine
            query = np.random.rand(128).astype(np.float32)
            params = SearchParams(k=10)
            
            results1 = db1.search(query, params)
            results2 = db2.search(query, params)
            
            for r1, r2 in zip(results1, results2):
                assert r1.id == r2.id
                assert abs(r1.distance - r2.distance) < 1e-5
    
    def test_file_not_exists(self):
        """Test loading from non-existent file raises error"""
        config = DatabaseConfig(
            dimension=32,
            index_type=IndexType.FLAT,
            metric=DistanceMetric.L2
        )
        
        db = SageDB(config)
        
        with pytest.raises(Exception):  # Should raise SageDBException
            db.load("/nonexistent/path/file.sagedb")
    
    def test_multiple_saves(self):
        """Test multiple save operations to same file"""
        config = DatabaseConfig(
            dimension=32,
            index_type=IndexType.FLAT,
            metric=DistanceMetric.L2
        )
        
        db = SageDB(config)
        vectors = np.random.rand(50, 32).astype(np.float32)
        db.add_vectors(vectors)
        db.build_index()
        
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "multi.sagedb")
            
            # Save multiple times
            db.save(filepath)
            db.save(filepath)  # Overwrite
            
            # Verify can still load
            db2 = SageDB(config)
            db2.load(filepath)
            assert db2.size() == 50


def test_persistence_example_runs():
    """Test that the example script can be imported and has expected functions"""
    import sys
    import os
    
    # Add examples directory to path
    examples_dir = os.path.join(os.path.dirname(__file__), "..", "examples")
    if os.path.exists(examples_dir):
        sys.path.insert(0, examples_dir)
        
        try:
            import python_persistence_example
            
            # Check expected functions exist
            assert hasattr(python_persistence_example, 'basic_persistence_example')
            assert hasattr(python_persistence_example, 'metadata_persistence_example')
            assert hasattr(python_persistence_example, 'incremental_save_example')
        except ImportError:
            pytest.skip("Example script not available")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

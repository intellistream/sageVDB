#!/usr/bin/env python3
"""
Test SageVDB persistence functionality

This test verifies that save/load operations work correctly
for both vector data and metadata.
"""

import pytest
import numpy as np
import os
import tempfile
from sagevdb import (
    SageVDB,
    DatabaseConfig,
    IndexType,
    DistanceMetric,
    SearchParams,
    add_numpy,
    search_numpy,
)


class TestPersistence:
    """Test persistence functionality"""
    
    def test_basic_save_load(self):
        """Test basic save and load operations"""
        config = DatabaseConfig(128)
        config.index_type = IndexType.FLAT
        config.metric = DistanceMetric.L2
        
        # Create and populate database
        db1 = SageVDB(config)
        vectors = np.random.rand(100, 128).astype(np.float32)
        ids = add_numpy(db1, vectors)
        db1.build_index()
        
        # Save to temporary file
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "test.SageVDB")
            db1.save(filepath)
            
            # Load into new database
            db2 = SageVDB(config)
            db2.load(filepath)
            
            # Verify size
            assert db2.size() == db1.size() == 100
            
            # Verify search results match
            query = np.random.rand(128).astype(np.float32)
            params = SearchParams(k=5)
            results1 = search_numpy(db1, query, params)
            results2 = search_numpy(db2, query, params)
            
            assert len(results1) == len(results2)
            for r1, r2 in zip(results1, results2):
                assert r1.id == r2.id
                assert abs(r1.score - r2.score) < 1e-5
    
    def test_metadata_persistence(self):
        """Test metadata save and load"""
        config = DatabaseConfig(64)
        config.index_type = IndexType.FLAT
        config.metric = DistanceMetric.COSINE
        
        db1 = SageVDB(config)
        
        # Add vectors with metadata
        metadata_map = {}
        for i in range(50):
            vec = np.random.rand(64).astype(np.float32)
            vec_id = db1.add(vec.tolist())
            meta = {"index": str(i), "category": f"cat_{i % 3}"}
            db1.set_metadata(vec_id, meta)
            metadata_map[vec_id] = meta
        
        db1.build_index()
        
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.SageVDB")
            meta_path = os.path.join(tmpdir, "test.meta")
            
            db1.save(db_path)
            db1.metadata_store().save(meta_path)
            
            # Load into new database
            db2 = SageVDB(config)
            db2.load(db_path)
            db2.metadata_store().load(meta_path)
            
            # Verify metadata
            for vec_id, expected_meta in metadata_map.items():
                actual_meta = db2.get_metadata(vec_id)
                assert actual_meta == expected_meta
    
    def test_empty_database_save_load(self):
        """Test save/load of empty database"""
        config = DatabaseConfig(32)
        config.index_type = IndexType.FLAT
        config.metric = DistanceMetric.L2
        
        db1 = SageVDB(config)
        db1.build_index()  # Build empty index
        
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "empty.SageVDB")
            db1.save(filepath)
            
            db2 = SageVDB(config)
            db2.load(filepath)
            
            assert db2.size() == 0
            assert db2.is_trained()
    
    def test_incremental_updates(self):
        """Test save -> load -> add more -> save again"""
        config = DatabaseConfig(64)
        config.index_type = IndexType.FLAT
        config.metric = DistanceMetric.L2
        
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "incremental.SageVDB")
            
            # Initial save
            db1 = SageVDB(config)
            vecs1 = np.random.rand(50, 64).astype(np.float32)
            add_numpy(db1, vecs1)
            db1.build_index()
            db1.save(filepath)
            
            # Load and add more
            db2 = SageVDB(config)
            db2.load(filepath)
            assert db2.size() == 50
            
            vecs2 = np.random.rand(30, 64).astype(np.float32)
            add_numpy(db2, vecs2)
            db2.build_index()
            db2.save(filepath)
            
            # Load final state
            db3 = SageVDB(config)
            db3.load(filepath)
            assert db3.size() == 80
    
    def test_different_index_types(self):
        """Test persistence works with different index types"""
        index_types = [IndexType.FLAT, IndexType.HNSW]
        
        for idx_type in index_types:
            config = DatabaseConfig(32)
            config.index_type = idx_type
            config.metric = DistanceMetric.L2
            
            if idx_type == IndexType.HNSW:
                config.M = 16
                config.efConstruction = 200
            
            db1 = SageVDB(config)
            vectors = np.random.rand(100, 32).astype(np.float32)
            add_numpy(db1, vectors)
            db1.build_index()
            
            with tempfile.TemporaryDirectory() as tmpdir:
                filepath = os.path.join(tmpdir, f"test_{idx_type}.SageVDB")
                db1.save(filepath)
                
                db2 = SageVDB(config)
                db2.load(filepath)
                
                assert db2.size() == 100
                assert db2.is_trained()
    
    def test_cosine_metric_persistence(self):
        """Test persistence with cosine similarity metric"""
        config = DatabaseConfig(128)
        config.index_type = IndexType.FLAT
        config.metric = DistanceMetric.COSINE
        
        db1 = SageVDB(config)
        vectors = np.random.rand(100, 128).astype(np.float32)
        add_numpy(db1, vectors)
        db1.build_index()
        
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "cosine.SageVDB")
            db1.save(filepath)
            
            db2 = SageVDB(config)
            db2.load(filepath)
            
            # Test search with cosine
            query = np.random.rand(128).astype(np.float32)
            params = SearchParams(k=10)
            
            results1 = search_numpy(db1, query, params)
            results2 = search_numpy(db2, query, params)
            
            for r1, r2 in zip(results1, results2):
                assert r1.id == r2.id
                assert abs(r1.score - r2.score) < 1e-5
    
    def test_file_not_exists(self):
        """Test loading from non-existent file raises error"""
        config = DatabaseConfig(32)
        config.index_type = IndexType.FLAT
        config.metric = DistanceMetric.L2
        
        db = SageVDB(config)
        
        with pytest.raises(Exception):  # Should raise SageVDBException
            db.load("/nonexistent/path/file.SageVDB")
    
    def test_multiple_saves(self):
        """Test multiple save operations to same file"""
        config = DatabaseConfig(32)
        config.index_type = IndexType.FLAT
        config.metric = DistanceMetric.L2
        
        db = SageVDB(config)
        vectors = np.random.rand(50, 32).astype(np.float32)
        add_numpy(db, vectors)
        db.build_index()
        
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "multi.SageVDB")
            
            # Save multiple times
            db.save(filepath)
            db.save(filepath)  # Overwrite
            
            # Verify can still load
            db2 = SageVDB(config)
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

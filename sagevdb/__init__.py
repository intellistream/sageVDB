"""
sage-vdb - High-Performance Vector Database with Pluggable ANNS Architecture
"""

try:
    from importlib.metadata import version, PackageNotFoundError
except ImportError:
    # Python < 3.8
    from importlib_metadata import version, PackageNotFoundError

try:
    __version__ = version("isage-vdb")
except PackageNotFoundError:
    # Package is not installed
    __version__ = "0.0.0+unknown"

__author__ = "IntelliStream Team"
__email__ = "shuhao_zhang@hust.edu.cn"

# Import the C++ extension module and expose all public classes
try:
    from ._sagevdb import (
        # Main classes
        SageVDB,
        VectorStore,
        MetadataStore,
        QueryEngine,
        
        # Configuration and parameters
        DatabaseConfig,
        SearchParams,
        
        # Enums
        IndexType,
        DistanceMetric,
        
        # Result types
        QueryResult,
        SearchStats,
        
        # Factory functions
        create_database,
        
        # Utility functions
        index_type_to_string,
        string_to_index_type,
        distance_metric_to_string,
        string_to_distance_metric,
        
        # NumPy helpers
        add_numpy,
        search_numpy,
        
        # Exception
        SageVDBException,
    )
    
    __all__ = [
        # Main classes
        'SageVDB',
        'VectorStore',
        'MetadataStore',
        'QueryEngine',
        
        # Configuration and parameters
        'DatabaseConfig',
        'SearchParams',
        
        # Enums
        'IndexType',
        'DistanceMetric',
        
        # Result types
        'QueryResult',
        'SearchStats',
        
        # Factory functions
        'create_database',
        
        # Utility functions
        'index_type_to_string',
        'string_to_index_type',
        'distance_metric_to_string',
        'string_to_distance_metric',
        
        # NumPy helpers
        'add_numpy',
        'search_numpy',
        
        # Exception
        'SageVDBException',
    ]
    
except ImportError as e:
    import warnings
    warnings.warn(
        f"Failed to import SageVDB native extension: {e}\n"
        "The package may not be properly installed. "
        "Try reinstalling with: pip install --force-reinstall SageVDB",
        ImportWarning
    )
    # Provide empty stubs to prevent total failure
    __all__ = []

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
        create_database as create_cpp_database,
        
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
        'create_cpp_database',
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

    try:
        from .sage_anns import SageANNSVectorStore, list_sage_anns_algorithms

        __all__.extend([
            'SageANNSVectorStore',
            'list_sage_anns_algorithms',
        ])
    except ImportError:
        SageANNSVectorStore = None
        list_sage_anns_algorithms = None

    def _metric_to_string(metric) -> str:
        if isinstance(metric, DistanceMetric):
            return distance_metric_to_string(metric)
        return str(metric)

    def create_database(*args, backend: str = "cpp", **kwargs):
        """Create a database instance.

        Args:
            backend: "cpp" for native SageVDB, "sage-anns" for Python ANNS backend.
        """
        backend_value = backend.lower().replace("_", "-")
        if backend_value in ("cpp", "native"):
            return create_cpp_database(*args, **kwargs)

        if backend_value not in ("sage-anns", "sageanns", "anns"):
            raise ValueError(
                "Unknown backend. Use 'cpp' or 'sage-anns'."
            )

        if SageANNSVectorStore is None:
            raise ImportError(
                "sage-anns backend requested but isage-anns is not installed. "
                "Install with: pip install isage-anns"
            )

        if len(args) == 1 and isinstance(args[0], DatabaseConfig):
            config = args[0]
            algorithm = kwargs.pop("algorithm", config.anns_algorithm)
            if not algorithm:
                raise ValueError("'algorithm' must be specified via DatabaseConfig.anns_algorithm or argument")
            params = dict(config.anns_build_params)
            params.update(kwargs)
            return SageANNSVectorStore(
                dimension=config.dimension,
                algorithm=algorithm,
                metric=_metric_to_string(config.metric),
                **params,
            )

        if len(args) >= 1:
            dimension = args[0]
            metric = kwargs.pop("metric", DistanceMetric.L2)
            algorithm = kwargs.pop("algorithm", kwargs.pop("anns_algorithm", None))
            if algorithm is None:
                raise ValueError("'algorithm' is required for sage-anns backend")
            return SageANNSVectorStore(
                dimension=dimension,
                algorithm=algorithm,
                metric=_metric_to_string(metric),
                **kwargs,
            )

        raise ValueError("Invalid arguments for create_database")
    
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

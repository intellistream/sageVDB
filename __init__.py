"""
sageDB - High-Performance Vector Database with Pluggable ANNS Architecture
"""

try:
    from importlib.metadata import version, PackageNotFoundError
except ImportError:
    # Python < 3.8
    from importlib_metadata import version, PackageNotFoundError

try:
    __version__ = version("sagedb")
except PackageNotFoundError:
    # Package is not installed
    __version__ = "0.0.0+unknown"

__author__ = "IntelliStream Team"
__email__ = "shuhao_zhang@hust.edu.cn"

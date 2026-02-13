cmake_minimum_required(VERSION 3.15)

# Find Python - use Development.Module for wheel builds (not Development.Embed)
find_package(Python3 COMPONENTS Interpreter Development.Module REQUIRED)

# Find or fetch pybind11 (check if parent project already provided it)
if(NOT TARGET pybind11::module)
    find_package(pybind11 CONFIG QUIET)
    
    if(NOT pybind11_FOUND)
        message(STATUS "pybind11 not found, fetching from GitHub...")
        include(FetchContent)
        FetchContent_Declare(
            pybind11
            GIT_REPOSITORY https://github.com/pybind/pybind11.git
            GIT_TAG v2.13.0
        )
        FetchContent_MakeAvailable(pybind11)
    endif()
else()
    message(STATUS "Using pybind11 provided by parent project")
endif()

# Create Python bindings module
pybind11_add_module(_sagevdb bindings.cpp)

# Link against SAGE VDB library
target_link_libraries(_sagevdb PRIVATE sage_vdb)

# Include directories
target_include_directories(_sagevdb PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../include
)

# Set module properties
set_target_properties(_sagevdb PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
    PREFIX "${PYTHON_MODULE_PREFIX}"
    SUFFIX "${PYTHON_MODULE_EXTENSION}"
    # SOTA: Use RPATH $ORIGIN to find libraries in same directory
    BUILD_RPATH_USE_ORIGIN TRUE
    INSTALL_RPATH "$ORIGIN"
    INSTALL_RPATH_USE_LINK_PATH TRUE
)

# Install the module
# SOTA: Conditional install based on build context
if(DEFINED SKBUILD)
    if(SKBUILD_STATE STREQUAL "editable")
        # Editable install: use parent-defined install dir if available, else fallback
        if(DEFINED sage_vdb_INSTALL_DIR)
            set(_install_dest "${sage_vdb_INSTALL_DIR}")
            message(STATUS "sage_vdb: Using parent-defined install dir: ${_install_dest}")
        else()
            set(_install_dest "${CMAKE_CURRENT_SOURCE_DIR}/../sagevdb")
            message(STATUS "sage_vdb: Using default editable install dir: ${_install_dest}")
        endif()
        # Also install C++ libraries to same location
        install(TARGETS sage_vdb
            LIBRARY DESTINATION ${_install_dest}
            ARCHIVE DESTINATION ${_install_dest}
            RUNTIME DESTINATION ${_install_dest}
            COMPONENT python
        )
    else()
        # Wheel build: install to wheel platlib
        set(_install_dest "${SKBUILD_PLATLIB_DIR}/sagevdb")
    endif()
    
    install(TARGETS _sagevdb
        LIBRARY DESTINATION ${_install_dest}
        COMPONENT python
    )
else()
    # Standalone C++ project build
    install(TARGETS _sagevdb
        LIBRARY DESTINATION python
        COMPONENT python
    )
endif()

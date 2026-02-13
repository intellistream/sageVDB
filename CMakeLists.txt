cmake_minimum_required(VERSION 3.12)
project(sage_vdb VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Build options
option(BUILD_TESTS "Build test programs" ON)
option(USE_OPENMP "Enable OpenMP support" ON)
option(ENABLE_MULTIMODAL "Enable multimodal fusion support" ON)
option(ENABLE_OPENCV "Enable OpenCV for image processing" OFF)
option(ENABLE_FFMPEG "Enable FFmpeg for audio/video processing" OFF)

set(_sage_vdb_enable_gperftools_default OFF)
if(DEFINED SAGE_ENABLE_GPERFTOOLS)
    set(_sage_vdb_enable_gperftools_default ${SAGE_ENABLE_GPERFTOOLS})
endif()
option(ENABLE_GPERFTOOLS "Enable gperftools profiling support" ${_sage_vdb_enable_gperftools_default})
if(DEFINED SAGE_ENABLE_GPERFTOOLS)
    set(ENABLE_GPERFTOOLS ${SAGE_ENABLE_GPERFTOOLS} CACHE BOOL "Enable gperftools profiling support" FORCE)
endif()

# Find packages (keep minimal to improve portability)

# Add cmake modules directory to the path
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

# Use enhanced BLAS/LAPACK finder
include(FindBLASLAPACK)

# Try to find FAISS
find_path(FAISS_INCLUDE_DIR NAMES faiss/IndexFlat.h
    HINTS
    ${FAISS_ROOT}/include
    /usr/local/include
    /usr/include
    /opt/conda/include
    $ENV{CONDA_PREFIX}/include
)

find_library(FAISS_LIBRARY NAMES faiss
    HINTS
    ${FAISS_ROOT}/lib
    /usr/local/lib
    /usr/lib
    /opt/conda/lib
    $ENV{CONDA_PREFIX}/lib
)

if(FAISS_INCLUDE_DIR AND FAISS_LIBRARY)
    message(STATUS "Found FAISS: ${FAISS_LIBRARY}")
    set(FAISS_FOUND TRUE)
else()
    message(STATUS "FAISS C++ library not found, using fallback implementation")
    message(STATUS "Note: Python faiss package should be available for full functionality")
    set(FAISS_FOUND FALSE)
endif()

# OpenMP support
if(USE_OPENMP)
    find_package(OpenMP)
    if(OpenMP_CXX_FOUND)
        message(STATUS "OpenMP found")
    endif()
endif()

# Source files
set(sage_vdb_SOURCES
    src/sage_vdb.cpp
    src/vector_store.cpp
    src/metadata_store.cpp
    src/query_engine.cpp
    src/anns/anns_interface.cpp
    src/anns/brute_force_plugin.cpp
)

set(sage_vdb_HEADERS
    include/sage_vdb/sage_vdb.h
    include/sage_vdb/vector_store.h
    include/sage_vdb/metadata_store.h
    include/sage_vdb/query_engine.h
    include/sage_vdb/common.h
    include/sage_vdb/anns/anns_interface.h
    include/sage_vdb/anns/brute_force_plugin.h
)


if(FAISS_FOUND)
    list(APPEND sage_vdb_SOURCES src/anns/faiss_plugin.cpp)
    list(APPEND sage_vdb_HEADERS include/sage_vdb/anns/faiss_plugin.h)
endif()

# Multimodal fusion sources and headers
if(ENABLE_MULTIMODAL)
    list(APPEND sage_vdb_SOURCES
        src/multimodal_sage_vdb.cpp
        src/fusion_strategies.cpp
        src/modality_manager.cpp
        src/modality_processors.cpp
    )
    
    list(APPEND sage_vdb_HEADERS
        include/sage_vdb/multimodal_fusion.h
        include/sage_vdb/multimodal_sage_vdb.h
        include/sage_vdb/fusion_strategies.h
        include/sage_vdb/modality_processors.h
    )
endif()

# Create shared library
add_library(sage_vdb SHARED ${sage_vdb_SOURCES})


# Include directories
target_include_directories(sage_vdb PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

if(FAISS_FOUND)
    target_include_directories(sage_vdb PRIVATE ${FAISS_INCLUDE_DIR})
    target_link_libraries(sage_vdb PRIVATE ${FAISS_LIBRARY})
    if(HAVE_BLAS_LAPACK)
        target_link_libraries(sage_vdb PRIVATE ${BLAS_LIBRARIES} ${LAPACK_LIBRARIES})
    endif()
    target_compile_definitions(sage_vdb PRIVATE ENABLE_FAISS)
endif()
# Compiler specific options
target_compile_options(sage_vdb PRIVATE
    $<$<CXX_COMPILER_ID:GNU>:-Wall -Wextra -O3>
    $<$<CXX_COMPILER_ID:Clang>:-Wall -Wextra -O3>
    $<$<CXX_COMPILER_ID:MSVC>:/W4 /O2>
)

# Tests
if(BUILD_TESTS)
    enable_testing()
    
    add_executable(test_sage_vdb tests/test_sage_vdb.cpp)
    target_link_libraries(test_sage_vdb PRIVATE sage_vdb)
    target_include_directories(test_sage_vdb PRIVATE include)
    
    add_test(NAME test_sage_vdb COMMAND test_sage_vdb)

    add_executable(test_anns_registry tests/test_anns_registry.cpp)
    target_link_libraries(test_anns_registry PRIVATE sage_vdb)
    target_include_directories(test_anns_registry PRIVATE include)

    add_test(NAME test_anns_registry COMMAND test_anns_registry)
    
    # Multimodal tests
    if(ENABLE_MULTIMODAL)
        add_executable(test_multimodal tests/test_multimodal.cpp)
        target_link_libraries(test_multimodal PRIVATE sage_vdb)
        target_include_directories(test_multimodal PRIVATE include)
        
        add_test(NAME test_multimodal COMMAND test_multimodal)
    endif()
    
endif()

# Install rules - 条件化安装以支持 SKBUILD 和独立构建
if(DEFINED SKBUILD)
    # 判断是否是 editable install
    # SKBUILD_STATE 可能是 "editable" 或 "wheel" 或未定义
    # 如果未定义或不是 "wheel"，默认按 editable 处理
    set(_is_editable TRUE)
    if(DEFINED SKBUILD_STATE)
        if(SKBUILD_STATE STREQUAL "wheel")
            set(_is_editable FALSE)
        endif()
    endif()
    
    if(_is_editable)
        # Editable install: use parent-defined install dir if available, else fallback
        if(DEFINED sage_vdb_INSTALL_DIR)
            set(_lib_install_dest "${sage_vdb_INSTALL_DIR}")
            message(STATUS "SageVDB: Using parent-defined install dir: ${_lib_install_dest}")
        else()
            set(_lib_install_dest "${CMAKE_CURRENT_SOURCE_DIR}/sagevdb")
            message(STATUS "SageVDB: Using default editable install dir: ${_lib_install_dest}")
        endif()
    else()
        # Wheel build: install to wheel platlib (same directory as Python extension)
        set(_lib_install_dest "${SKBUILD_PLATLIB_DIR}/sagevdb")
        message(STATUS "SageVDB: Wheel build, lib -> ${_lib_install_dest}")
    endif()
    
    install(TARGETS sage_vdb
        LIBRARY DESTINATION ${_lib_install_dest}
        ARCHIVE DESTINATION ${_lib_install_dest}
        RUNTIME DESTINATION ${_lib_install_dest}
        COMPONENT python
    )
else()
    # 独立 C++ 构建模式：标准安装路径
    install(TARGETS sage_vdb
        EXPORT sage_vdb_targets
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib
        RUNTIME DESTINATION bin
    )
endif()

install(DIRECTORY include/ DESTINATION include)

# ============================================================================
# Python bindings (optional)
# ============================================================================
option(BUILD_PYTHON_BINDINGS "Build Python bindings" OFF)

if(BUILD_PYTHON_BINDINGS)
    message(STATUS "Building Python bindings for sage_vdb")
    add_subdirectory(python)
endif()

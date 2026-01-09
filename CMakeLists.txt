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
option(ENABLE_SONG "Enable SONG GPU ANN backend" OFF)
option(ENABLE_FLATGPU_CUDA "Enable CUDA acceleration for FlatGPU index" OFF)
option(ENABLE_LIBAMM "Enable LibAMM accelerated sketch backends" ON)

set(_sage_db_enable_gperftools_default OFF)
if(DEFINED SAGE_ENABLE_GPERFTOOLS)
    set(_sage_db_enable_gperftools_default ${SAGE_ENABLE_GPERFTOOLS})
endif()
option(ENABLE_GPERFTOOLS "Enable gperftools profiling support" ${_sage_db_enable_gperftools_default})
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
set(SAGE_DB_SOURCES
    src/sage_db.cpp
    src/vector_store.cpp
    src/metadata_store.cpp
    src/query_engine.cpp
    src/anns/anns_interface.cpp
    src/anns/brute_force_plugin.cpp
)

set(SAGE_DB_HEADERS
    include/sage_db/sage_db.h
    include/sage_db/vector_store.h
    include/sage_db/metadata_store.h
    include/sage_db/query_engine.h
    include/sage_db/common.h
    include/sage_db/anns/anns_interface.h
    include/sage_db/anns/brute_force_plugin.h
)

if(ENABLE_SONG)
    message(STATUS "SONG ANN backend enabled")
    list(APPEND SAGE_DB_SOURCES src/anns/song_plugin.cpp)
    list(APPEND SAGE_DB_HEADERS include/sage_db/anns/song_plugin.h)
endif()

list(APPEND SAGE_DB_SOURCES src/anns/vamana_plugin.cpp)
list(APPEND SAGE_DB_HEADERS include/sage_db/anns/vamana_plugin.h)

list(APPEND SAGE_DB_SOURCES src/anns/flat_gpu_plugin.cpp)
list(APPEND SAGE_DB_HEADERS include/sage_db/anns/flat_gpu_plugin.h)
list(APPEND SAGE_DB_HEADERS include/sage_db/anns/flat_gpu/cuda_helpers.h)

if(ENABLE_FLATGPU_CUDA)
    enable_language(CUDA)
    list(APPEND SAGE_DB_SOURCES src/anns/flat_gpu/cuda_helpers.cu)
else()
    list(APPEND SAGE_DB_SOURCES src/anns/flat_gpu/cuda_helpers_stub.cpp)
endif()

if(FAISS_FOUND)
    list(APPEND SAGE_DB_SOURCES src/anns/faiss_plugin.cpp)
    list(APPEND SAGE_DB_HEADERS include/sage_db/anns/faiss_plugin.h)
endif()

# Multimodal fusion sources and headers
if(ENABLE_MULTIMODAL)
    list(APPEND SAGE_DB_SOURCES
        src/multimodal_sage_db.cpp
        src/fusion_strategies.cpp
        src/modality_manager.cpp
        src/modality_processors.cpp
    )
    
    list(APPEND SAGE_DB_HEADERS
        include/sage_db/multimodal_fusion.h
        include/sage_db/multimodal_sage_db.h
        include/sage_db/fusion_strategies.h
        include/sage_db/modality_processors.h
    )
endif()

# Create shared library
add_library(sage_db SHARED ${SAGE_DB_SOURCES})

if(ENABLE_SONG)
    target_compile_definitions(sage_db PRIVATE ENABLE_SONG)
endif()

if(ENABLE_FLATGPU_CUDA)
    target_compile_definitions(sage_db PRIVATE ENABLE_FLATGPU_CUDA)
endif()

# Include directories
target_include_directories(sage_db PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

if(FAISS_FOUND)
    target_include_directories(sage_db PRIVATE ${FAISS_INCLUDE_DIR})
    target_link_libraries(sage_db PRIVATE ${FAISS_LIBRARY})
    if(HAVE_BLAS_LAPACK)
        target_link_libraries(sage_db PRIVATE ${BLAS_LIBRARIES} ${LAPACK_LIBRARIES})
    endif()
    target_compile_definitions(sage_db PRIVATE ENABLE_FAISS)
endif()

# Multimodal fusion dependencies
if(ENABLE_MULTIMODAL)
    target_compile_definitions(sage_db PRIVATE MULTIMODAL_ENABLED)
    
    # OpenCV for image processing
    if(ENABLE_OPENCV)
        find_package(OpenCV REQUIRED)
        target_include_directories(sage_db PRIVATE ${OpenCV_INCLUDE_DIRS})
        target_link_libraries(sage_db PRIVATE ${OpenCV_LIBS})
        target_compile_definitions(sage_db PRIVATE OPENCV_ENABLED)
    endif()
    
    # FFmpeg for audio/video processing  
    if(ENABLE_FFMPEG)
        find_package(PkgConfig)
        if(PKG_CONFIG_FOUND)
            pkg_check_modules(FFMPEG REQUIRED libavformat libavcodec libavutil libswscale)
            target_include_directories(sage_db PRIVATE ${FFMPEG_INCLUDE_DIRS})
            target_link_libraries(sage_db PRIVATE ${FFMPEG_LIBRARIES})
            target_compile_definitions(sage_db PRIVATE FFMPEG_ENABLED)
        endif()
    endif()
endif()

# OpenMP linking
if(OpenMP_CXX_FOUND)
    target_link_libraries(sage_db PUBLIC OpenMP::OpenMP_CXX)
endif()

# LibAMM accelerated sketch support
if(ENABLE_LIBAMM)
    # Attempt to locate Torch (required by LibAMM)
    # PyTorch's CMake config fails with FATAL_ERROR if it's CUDA-enabled but CUDA is not found
    # We check this condition before calling find_package to avoid the error
    
    set(_torch_usable FALSE)
    
    # First, check if PyTorch is available via Python and if it requires CUDA
    find_package(Python3 COMPONENTS Interpreter QUIET)
    if(Python3_Interpreter_FOUND)
        # Check if torch is available and if it's CUDA-enabled
        execute_process(
            COMMAND ${Python3_EXECUTABLE} -c "import sys; import torch; sys.stdout.write('1' if torch.cuda.is_available() else '0'); sys.stdout.write('|'); sys.stdout.write(torch.version.cuda if hasattr(torch.version, 'cuda') and torch.version.cuda else 'none')"
            OUTPUT_VARIABLE _Torch_CUDA_Info
            RESULT_VARIABLE _Torch_Check_Result
            ERROR_QUIET
            OUTPUT_STRIP_TRAILING_WHITESPACE
        )
        
        if(_Torch_Check_Result EQUAL 0)
            string(REPLACE "|" ";" _Torch_CUDA_List "${_Torch_CUDA_Info}")
            list(GET _Torch_CUDA_List 0 _Torch_Has_CUDA_Runtime)
            list(GET _Torch_CUDA_List 1 _Torch_CUDA_Version)
            
            # If PyTorch is CUDA-enabled but CUDA runtime is not available, skip Torch finding
            if(NOT _Torch_CUDA_Version STREQUAL "none" AND _Torch_Has_CUDA_Runtime STREQUAL "0")
                message(WARNING "PyTorch is CUDA-enabled (version ${_Torch_CUDA_Version}) but CUDA runtime is not available")
                message(WARNING "Disabling LibAMM to avoid CMake configuration errors")
                message(WARNING "To enable LibAMM:")
                message(WARNING "  - Install CUDA toolkit matching PyTorch CUDA version ${_Torch_CUDA_Version}")
                message(WARNING "  - Or install CPU-only PyTorch: pip install torch --index-url https://download.pytorch.org/whl/cpu")
                set(ENABLE_LIBAMM OFF CACHE BOOL "Enable LibAMM accelerated sketch backends" FORCE)
                set(_torch_usable FALSE)
            else()
                # PyTorch is CPU-only or CUDA is available
                set(_torch_usable TRUE)
            endif()
        else()
            message(WARNING "PyTorch not found in Python environment")
            set(ENABLE_LIBAMM OFF CACHE BOOL "Enable LibAMM accelerated sketch backends" FORCE)
            set(_torch_usable FALSE)
        endif()
    else()
        message(WARNING "Python interpreter not found, cannot check PyTorch availability")
        set(ENABLE_LIBAMM OFF CACHE BOOL "Enable LibAMM accelerated sketch backends" FORCE)
        set(_torch_usable FALSE)
    endif()

    # Only attempt to find Torch via CMake if we've verified it's usable
    if(_torch_usable AND NOT Torch_FOUND)
        # Get Torch CMake prefix path from Python
        execute_process(
            COMMAND ${Python3_EXECUTABLE} -c "import torch; print(torch.utils.cmake_prefix_path)"
            OUTPUT_VARIABLE _Torch_CMake_Prefix
            RESULT_VARIABLE _Torch_CMake_Result
            ERROR_QUIET
            OUTPUT_STRIP_TRAILING_WHITESPACE
        )
        
        if(_Torch_CMake_Result EQUAL 0 AND NOT _Torch_CMake_Prefix STREQUAL "")
            list(APPEND CMAKE_PREFIX_PATH "${_Torch_CMake_Prefix}")
            find_package(Torch QUIET)
            
            if(Torch_FOUND)
                message(STATUS "PyTorch found and loaded successfully for LibAMM support")
            else()
                message(WARNING "PyTorch CMake config not found despite Python detection")
                set(ENABLE_LIBAMM OFF CACHE BOOL "Enable LibAMM accelerated sketch backends" FORCE)
            endif()
        else()
            message(WARNING "Could not determine PyTorch CMake prefix path")
            set(ENABLE_LIBAMM OFF CACHE BOOL "Enable LibAMM accelerated sketch backends" FORCE)
        endif()
    endif()
endif()

if(ENABLE_LIBAMM)
    message(STATUS "LibAMM support enabled")

    # Check for local LibAMM installation first (using project-level external/ directory)
    set(LIBAMM_LOCAL_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../../../../../../../../../external/LibAMM" CACHE PATH "Path to local LibAMM installation")
    
    if(EXISTS "${LIBAMM_LOCAL_PATH}/CMakeLists.txt")
        message(STATUS "Using local LibAMM from: ${LIBAMM_LOCAL_PATH}")
        set(libamm_SOURCE_DIR "${LIBAMM_LOCAL_PATH}")
        set(libamm_BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}/_deps/libamm-build")
        
        set(_saved_cxx_flags "${CMAKE_CXX_FLAGS}")
        set(_saved_cxx_flags_debug "${CMAKE_CXX_FLAGS_DEBUG}")
        set(_saved_cxx_flags_release "${CMAKE_CXX_FLAGS_RELEASE}")

        # Disable optional LibAMM extras we do not need when embedding as a subproject
        set(ENABLE_UNIT_TESTS OFF CACHE BOOL "Disable LibAMM unit tests" FORCE)
        set(ENABLE_PYBIND OFF CACHE BOOL "Disable LibAMM standalone pybind module" FORCE)
        set(ENABLE_PAPI OFF CACHE BOOL "Disable LibAMM PAPI instrumentation" FORCE)
        set(ENABLE_OPENCL OFF CACHE BOOL "Disable LibAMM OpenCL backend" FORCE)
        set(BUILD_BENCHMARKS OFF CACHE BOOL "Disable LibAMM benchmarks" FORCE)
        
        # Set as regular variable first (LibAMM checks non-cached variable)
        set(LibAMM_SKIP_BENCHMARK_COPY ON)
        set(LibAMM_SKIP_BENCHMARK_COPY ON CACHE BOOL "Skip LibAMM benchmark dataset copy" FORCE)
        
        add_subdirectory(${libamm_SOURCE_DIR} ${libamm_BINARY_DIR} EXCLUDE_FROM_ALL)
        
        set(CMAKE_CXX_FLAGS "${_saved_cxx_flags}")
        set(CMAKE_CXX_FLAGS_DEBUG "${_saved_cxx_flags_debug}")
        set(CMAKE_CXX_FLAGS_RELEASE "${_saved_cxx_flags_release}")
    else()
        message(STATUS "Local LibAMM not found at ${LIBAMM_LOCAL_PATH}")
        message(STATUS "Attempting to download LibAMM via FetchContent...")
        message(STATUS "Note: For faster builds, you can manually download LibAMM to ${LIBAMM_LOCAL_PATH}")
        
        include(FetchContent)
        
        # Use URL download as fallback if git clone fails (more reliable for users)
        set(LIBAMM_USE_URL_DOWNLOAD OFF CACHE BOOL "Download LibAMM as tarball instead of git clone")
        set(LIBAMM_GIT_TAG "feat/configurable-dataset-path" CACHE STRING "LibAMM git branch/tag to use")
        
        if(LIBAMM_USE_URL_DOWNLOAD)
            FetchContent_Declare(
                libamm
                URL https://github.com/intellistream/LibAMM/archive/refs/heads/${LIBAMM_GIT_TAG}.tar.gz
                DOWNLOAD_EXTRACT_TIMESTAMP TRUE
            )
        else()
            FetchContent_Declare(
                libamm
                GIT_REPOSITORY https://github.com/intellistream/LibAMM.git
                GIT_TAG ${LIBAMM_GIT_TAG}
                GIT_PROGRESS TRUE
            )
        endif()

        set(_saved_cxx_flags "${CMAKE_CXX_FLAGS}")
        set(_saved_cxx_flags_debug "${CMAKE_CXX_FLAGS_DEBUG}")
        set(_saved_cxx_flags_release "${CMAKE_CXX_FLAGS_RELEASE}")

        # Disable optional LibAMM extras we do not need when embedding as a subproject
        # These must be set BEFORE FetchContent_MakeAvailable to take effect
        set(ENABLE_UNIT_TESTS OFF CACHE BOOL "Disable LibAMM unit tests" FORCE)
        set(ENABLE_PYBIND OFF CACHE BOOL "Disable LibAMM standalone pybind module" FORCE)
        set(ENABLE_PAPI OFF CACHE BOOL "Disable LibAMM PAPI instrumentation" FORCE)
        set(ENABLE_OPENCL OFF CACHE BOOL "Disable LibAMM OpenCL backend" FORCE)
        set(BUILD_BENCHMARKS OFF CACHE BOOL "Disable LibAMM benchmarks" FORCE)
        
        # Skip LibAMM benchmark dataset copy (we don't need datasets for library usage)
        # Note: LibAMM benchmark datasets are now centrally managed in sageData repository
        # If you need to run LibAMM benchmarks, set:
        #   -DLIBAMM_SKIP_DATASET_COPY=OFF
        #   -DLIBAMM_DATASET_SOURCE_DIR="/path/to/sageData/libamm-benchmark/datasets"
        set(LIBAMM_SKIP_DATASET_COPY ON CACHE BOOL "Skip LibAMM benchmark dataset copy" FORCE)

        FetchContent_MakeAvailable(libamm)

        # Restore original compiler flags to prevent LibAMM from overriding the parent project
        set(CMAKE_CXX_FLAGS "${_saved_cxx_flags}")
        set(CMAKE_CXX_FLAGS_DEBUG "${_saved_cxx_flags_debug}")
        set(CMAKE_CXX_FLAGS_RELEASE "${_saved_cxx_flags_release}")
        
        # Note: FetchContent_MakeAvailable automatically sets libamm_SOURCE_DIR and libamm_BINARY_DIR
        # Downloaded to: ${CMAKE_CURRENT_BINARY_DIR}/_deps/libamm-src/
        # Built in: ${CMAKE_CURRENT_BINARY_DIR}/_deps/libamm-build/
    endif()

    target_link_libraries(sage_db PRIVATE LibAMM)

    if(Torch_FOUND)
        if(TORCH_INCLUDE_DIRS)
            target_include_directories(sage_db PRIVATE ${TORCH_INCLUDE_DIRS})
        endif()
        if(TORCH_LIBRARIES)
            target_link_libraries(sage_db PRIVATE ${TORCH_LIBRARIES})
        endif()
        if(TORCH_CXX_FLAGS)
            separate_arguments(_TORCH_CXX_FLAGS_LIST NATIVE_COMMAND ${TORCH_CXX_FLAGS})
            if(_TORCH_CXX_FLAGS_LIST)
                target_compile_options(sage_db PRIVATE ${_TORCH_CXX_FLAGS_LIST})
            endif()
        endif()
    endif()

    # Use libamm_SOURCE_DIR (set by either local path or FetchContent)
    target_include_directories(sage_db PRIVATE
        ${libamm_SOURCE_DIR}/include
    )
    target_compile_definitions(sage_db PRIVATE ENABLE_LIBAMM)
endif()

# Compiler specific options
target_compile_options(sage_db PRIVATE
    $<$<CXX_COMPILER_ID:GNU>:-Wall -Wextra -O3>
    $<$<CXX_COMPILER_ID:Clang>:-Wall -Wextra -O3>
    $<$<CXX_COMPILER_ID:MSVC>:/W4 /O2>
)

# Tests
if(BUILD_TESTS)
    enable_testing()
    
    add_executable(test_sage_db tests/test_sage_db.cpp)
    target_link_libraries(test_sage_db PRIVATE sage_db)
    target_include_directories(test_sage_db PRIVATE include)
    
    add_test(NAME test_sage_db COMMAND test_sage_db)
    
    # Multimodal tests
    if(ENABLE_MULTIMODAL)
        add_executable(test_multimodal tests/test_multimodal.cpp)
        target_link_libraries(test_multimodal PRIVATE sage_db)
        target_include_directories(test_multimodal PRIVATE include)
        
        add_test(NAME test_multimodal COMMAND test_multimodal)
    endif()
    
    # LibAMM integration tests
    if(ENABLE_LIBAMM)
        add_executable(test_libamm_integration tests/test_libamm_integration.cpp)
        target_link_libraries(test_libamm_integration PRIVATE sage_db)
        target_include_directories(test_libamm_integration PRIVATE include)
        
        add_test(NAME test_libamm_integration COMMAND test_libamm_integration)
        message(STATUS "LibAMM integration tests enabled")
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
        if(DEFINED SAGE_DB_INSTALL_DIR)
            set(_lib_install_dest "${SAGE_DB_INSTALL_DIR}")
            message(STATUS "sageDB: Using parent-defined install dir: ${_lib_install_dest}")
        else()
            set(_lib_install_dest "${CMAKE_CURRENT_SOURCE_DIR}/python")
            message(STATUS "sageDB: Using default editable install dir: ${_lib_install_dest}")
        endif()
    else()
        # Wheel build: install to wheel platlib (same directory as Python extension)
        set(_lib_install_dest "${SKBUILD_PLATLIB_DIR}/sagevdb")
        message(STATUS "sageDB: Wheel build, lib -> ${_lib_install_dest}")
    endif()
    
    install(TARGETS sage_db
        LIBRARY DESTINATION ${_lib_install_dest}
        ARCHIVE DESTINATION ${_lib_install_dest}
        RUNTIME DESTINATION ${_lib_install_dest}
        COMPONENT python
    )
else()
    # 独立 C++ 构建模式：标准安装路径
    install(TARGETS sage_db
        EXPORT sage_db_targets
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
    message(STATUS "Building Python bindings for sage_db")
    add_subdirectory(python)
endif()

# Resolve gperftools libraries when building sage_vdb stand-alone.
# This script assumes the boolean option ENABLE_GPERFTOOLS controls whether we attempt lookup.

if(NOT ENABLE_GPERFTOOLS)
    message(STATUS "gperftools support disabled for sage_vdb")
    set(SAGE_GPERFTOOLS_LIBS "")
    return()
endif()

find_library(SAGE_GPERFTOOLS_PROFILER_LIB profiler)
find_library(SAGE_GPERFTOOLS_TCMALLOC_LIB tcmalloc)

if(SAGE_GPERFTOOLS_PROFILER_LIB AND SAGE_GPERFTOOLS_TCMALLOC_LIB)
    set(SAGE_GPERFTOOLS_LIBS ${SAGE_GPERFTOOLS_PROFILER_LIB} ${SAGE_GPERFTOOLS_TCMALLOC_LIB})
    message(STATUS "gperftools found for sage_vdb: ${SAGE_GPERFTOOLS_LIBS}")
else()
    if(NOT SAGE_GPERFTOOLS_PROFILER_LIB)
        message(WARNING "Requested gperftools but libprofiler not found; disabling ENABLE_GPERFTOOLS")
    elseif(NOT SAGE_GPERFTOOLS_TCMALLOC_LIB)
        message(WARNING "Requested gperftools but libtcmalloc not found; disabling ENABLE_GPERFTOOLS")
    endif()
    set(SAGE_GPERFTOOLS_LIBS "")
    set(ENABLE_GPERFTOOLS OFF CACHE BOOL "Enable gperftools profiling support" FORCE)
endif()

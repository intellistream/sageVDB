# Standalone pybind11 dependency resolver for sage_vdb.
# When the build is orchestrated by the Python superbuild, a shared dependency
# file can be provided via SAGE_COMMON_DEPS_FILE. Otherwise this fallback keeps
# the project self-contained.

if(NOT TARGET pybind11::module)
    find_package(pybind11 CONFIG QUIET)
    if(pybind11_FOUND)
        message(STATUS "Using system pybind11 ${pybind11_VERSION} from ${pybind11_CONFIG}")
    else()
        include(FetchContent)
        if(NOT DEFINED pybind11_POPULATED)
            set(_sage_vdb_pybind11_tag "v2.13.0" CACHE STRING "Pinned pybind11 version for sage_vdb")
            FetchContent_Declare(
                pybind11
                GIT_REPOSITORY https://github.com/pybind/pybind11.git
                GIT_TAG ${_sage_vdb_pybind11_tag}
            )
        endif()
        FetchContent_MakeAvailable(pybind11)
        message(STATUS "Fetched pybind11 ${_sage_vdb_pybind11_tag} for sage_vdb")
    endif()
endif()

if(NOT COMMAND pybind11_add_module)
    message(FATAL_ERROR "pybind11_add_module is unavailable even after resolving pybind11 dependency")
endif()

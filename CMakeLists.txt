cmake_minimum_required(VERSION 3.20.1) # Minimum required for CUDA

# Set the project name and version
project(occupancy VERSION 1.0 LANGUAGES CXX)

# C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Kokkos and APEX should already be cloned via git submodule
set(Kokkos_ROOT ${PROJECT_SOURCE_DIR}/kokkos)
set(APEX_ROOT ${PROJECT_SOURCE_DIR}/apex)

# Add Kokkos and APEX to the build
add_subdirectory(kokkos)
add_subdirectory(apex)

# Enable testing and add test directory
enable_testing()
add_subdirectory(tests)

# Debug/info output
message(STATUS "----------------------------------------------------------------------")
message(STATUS "Variable Report:")
message(STATUS "----------------------------------------------------------------------")
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "Compiler CXX debug flags: ${CMAKE_CXX_FLAGS_DEBUG}")
message(STATUS "Compiler CXX release flags: ${CMAKE_CXX_FLAGS_RELEASE}")
message(STATUS "Compiler CXX min size flags: ${CMAKE_CXX_FLAGS_MINSIZEREL}")
message(STATUS "Compiler CXX flags: ${CMAKE_CXX_FLAGS}")
message(STATUS "Install Prefix: ${CMAKE_INSTALL_PREFIX}")
message(STATUS "----------------------------------------------------------------------")

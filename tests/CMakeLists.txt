# tests/CMakeLists.txt

add_executable(test_occupancy test_occupancy.cpp)

target_include_directories(test_occupancy PRIVATE
    ${PROJECT_SOURCE_DIR}/kokkos/include
    ${PROJECT_SOURCE_DIR}/apex/src
)

target_link_libraries(test_occupancy
    PRIVATE kokkoscore apex
)

target_compile_options(test_occupancy PRIVATE -Wall -Wextra -Wpedantic)

add_test(NAME OccupancyTest COMMAND test_occupancy)

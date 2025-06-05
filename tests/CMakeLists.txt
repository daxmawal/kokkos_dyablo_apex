# tests/CMakeLists.txt

add_executable(standalone_foreach_cell_chunk standalone_foreach_cell_chunk.cpp)
target_include_directories(standalone_foreach_cell_chunk PRIVATE
    ${PROJECT_SOURCE_DIR}/kokkos/include
    ${PROJECT_SOURCE_DIR}/apex/src
)
target_link_libraries(standalone_foreach_cell_chunk
    PRIVATE kokkoscore apex
)
target_compile_options(standalone_foreach_cell_chunk PRIVATE -Wall -Wextra -Wpedantic)
add_test(NAME OccupancyTest COMMAND standalone_foreach_cell_chunk)

add_executable(standalone_foreach_cell_policy standalone_foreach_cell_policy.cpp)

target_include_directories(standalone_foreach_cell_policy PRIVATE
    ${PROJECT_SOURCE_DIR}/kokkos/include
    ${PROJECT_SOURCE_DIR}/apex/src
)

target_link_libraries(standalone_foreach_cell_policy
    PRIVATE kokkoscore apex
)

target_compile_options(standalone_foreach_cell_policy PRIVATE -Wall -Wextra -Wpedantic)

add_test(NAME ForeachCellTuningTest COMMAND standalone_foreach_cell_policy)

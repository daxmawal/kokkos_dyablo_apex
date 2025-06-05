# Kokkos Dyablo APEX
Project to test Kokkos kernels instrumented with APEX to measure performance, in particular GPU occupancy and execution efficiency.

## Compilation
```bash
git submodule update --init --recursive
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
```

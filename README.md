# Kokkos Dyablo APEX
Project to test Kokkos kernels instrumented with APEX to measure performance, in particular GPU occupancy and execution efficiency.

## Compilation
```bash
git submodule update --init --recursive

cmake -B build \
-DCMAKE_BUILD_TYPE=RelWithDebInfo \
-DCMAKE_CXX_COMPILER=g++ \
-DCMAKE_C_COMPILER=gcc \
-DCMAKE_INSTALL_PREFIX=`pwd`/install \
-DKokkos_ENABLE_TUNING=ON \
-DKokkos_ENABLE_OPENMP=ON \
-DKokkos_ENABLE_SERIAL=ON \
-DKokkos_ENABLE_CUDA=ON \
-DKokkos_ENABLE_CUDA_LAMBDA=ON \
-DKokkos_ARCH_NATIVE=ON \
-DAPEX_WITH_CUDA=TRUE \
-DCUDAToolkit_ROOT=${CUDA} \
.

cmake --build build --parallel 16

cmake --build build --parallel --target install

```

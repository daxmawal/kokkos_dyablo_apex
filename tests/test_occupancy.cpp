#include <Kokkos_Core.hpp>
#include <iostream>

int
main(int argc, char* argv[])
{
  Kokkos::initialize(argc, argv);
  {
    const int N = 10;

    Kokkos::parallel_for(
        "InitLoop", N,
        KOKKOS_LAMBDA(const int i) { printf("Hello from iteration %d\n", i); });

    Kokkos::fence();
    std::cout << "Test completed successfully.\n";
  }
  Kokkos::finalize();
  return 0;
}
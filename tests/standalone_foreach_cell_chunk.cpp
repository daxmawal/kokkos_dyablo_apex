#include <Kokkos_Core.hpp>
#include <cstdint>
#include <iostream>
#include <string>

#include "tuning_playground.hpp"

struct CellArray_shape {
  uint32_t bx, by, bz;
};

struct CellIndex {
  struct {
    uint32_t octant;
    bool dummy;
  } octant;
  uint32_t i, j, k;
  uint32_t bx, by, bz;
};

struct MeshWrapper {
  uint32_t getNumOctants() const { return 20; }
};

// Classe isolée avec la fonction foreach_cell
struct CellExecutor {
  MeshWrapper pmesh;

  template <typename Function>
  void foreach_cell(
      const std::string& kernel_name, const CellArray_shape& iter_space,
      const Function& f) const
  {
    uint32_t bx = iter_space.bx;
    uint32_t by = iter_space.by;
    uint32_t bz = iter_space.bz;
    uint32_t nbCellsPerBlock = bx * by * bz;
    uint32_t nbOcts = pmesh.getNumOctants();
    uint32_t totalCells = nbCellsPerBlock * nbOcts;

    // Partie auto-tuning : déclarer chunk_size une seule fois (statique)
    static size_t chunk_size_id = []() {
      using namespace Kokkos::Tools::Experimental;
      VariableInfo info;
      info.type = ValueType::kokkos_value_int64;
      info.category = StatisticalCategory::kokkos_value_categorical;
      info.valueQuantity = CandidateValueType::kokkos_value_set;

      int64_t chunk_options[] = {8, 16, 32, 64, 128, 256, 512, 1024};
      info.candidates = make_candidate_set(8, chunk_options);
      return declare_output_type("foreach_cell.chunk_size", info);
    }();

    // Demander la valeur optimale de chunk_size au tuner
    using namespace Kokkos::Tools::Experimental;
    auto context_id = get_new_context_id();
    begin_context(context_id);

    int64_t default_chunk_size = 128;
    VariableValue chunk_size_value =
        make_variable_value(chunk_size_id, default_chunk_size);
    request_output_values(context_id, 1, &chunk_size_value);

    uint32_t chunk_size =
        static_cast<uint32_t>(chunk_size_value.value.int_value);
    std::cout << "[TUNING] Using chunk_size = " << chunk_size << "\n";
    end_context(context_id);

    // Appliquer la chunk_size au RangePolicy
    Kokkos::RangePolicy<> policy(0, totalCells);
    policy.set_chunk_size(chunk_size);

    // Lancer le kernel avec chunk size tuné
    Kokkos::parallel_for(
        kernel_name, policy, KOKKOS_LAMBDA(uint32_t index) {
          uint32_t iOct = index / nbCellsPerBlock;
          index = index % nbCellsPerBlock;

          uint32_t k = index / (bx * by);
          uint32_t j = (index - k * bx * by) / bx;
          uint32_t i = index - j * bx - k * bx * by;

          CellIndex iCell = {{iOct, false}, i, j, k, bx, by, bz};
          f(iCell, iOct * nbCellsPerBlock + index);
        });
  }
};

int
main(int argc, char* argv[])
{
  Kokkos::initialize(argc, argv);
  {
    Kokkos::print_configuration(std::cout, false);

    CellExecutor exec;
    CellArray_shape shape{128, 128, 128};
    uint32_t nbCells = shape.bx * shape.by * shape.bz * 20;

    Kokkos::View<int*> result("cell_sums", nbCells);

    // On répète 1000 fois pour permettre au tuner de converger
    constexpr int num_iterations = 1000;

    for (int iter = 0; iter < num_iterations; ++iter) {
      exec.foreach_cell(
          "foreach_demo", shape,
          KOKKOS_LAMBDA(CellIndex iCell, uint32_t flat_index) {
            result(flat_index) = iCell.i + iCell.j + iCell.k;
          });

      // Optionnel : fence après chaque itération (utile pour APEX)
      Kokkos::fence();
    }

    // Copier les premiers résultats sur le host après la dernière itération
    auto result_host =
        Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), result);

    std::cout << "Premiers résultats de i+j+k pour les premières cellules:\n";
    for (int i = 0; i < 10 && i < nbCells; ++i) {
      std::cout << "Cell " << i << " -> sum = " << result_host(i) << "\n";
    }

    Kokkos::fence();
  }
  Kokkos::finalize();
  return 0;
}
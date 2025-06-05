#include <Kokkos_Core.hpp>
#include <Kokkos_Profiling_ScopedRegion.hpp>
#include <iostream>
#include <string>

#include "tuning_playground.hpp"

// Définition de la forme d'un bloc de cellules
struct CellArray_shape {
  uint32_t bx, by, bz;
};

// Représentation d'une cellule dans un octant
struct CellIndex {
  struct {
    uint32_t octant;
    bool dummy;
  } octant;
  uint32_t i, j, k;
  uint32_t bx, by, bz;
};

// Simule un maillage avec un nombre fixe d'octants
struct MeshWrapper {
  uint32_t getNumOctants() const { return 20; }
};

struct CellExecutor {
  MeshWrapper pmesh;

  template <typename Function>
  void foreach_cell(
      const std::string& kernel_name, const CellArray_shape& shape,
      const Function& f) const
  {
    const uint32_t bx = shape.bx;
    const uint32_t by = shape.by;
    const uint32_t bz = shape.bz;
    const uint32_t cells_per_block = bx * by * bz;
    const uint32_t num_octants = pmesh.getNumOctants();
    const uint32_t total_cells = cells_per_block * num_octants;

    // Choix de politique via tuner
    static size_t policy_choice_id =
        create_categorical_int_tuner("foreach_cell.policy", 3);
    auto context_id = Kokkos::Tools::Experimental::get_new_context_id();
    Kokkos::Tools::Experimental::begin_context(context_id);
    Kokkos::Tools::Experimental::VariableValue policy_val =
        Kokkos::Tools::Experimental::make_variable_value(
            policy_choice_id, int64_t(0));
    Kokkos::Tools::Experimental::request_output_values(
        context_id, 1, &policy_val);
    int policy = policy_val.value.int_value;
    Kokkos::Tools::Experimental::end_context(context_id);

    Kokkos::Profiling::ScopedRegion region("foreach_cell_search_loop");

    for (int iter = 0; iter < Impl::max_iterations; ++iter) {
      // Teste différentes politiques d'exécution
      fastest_of(
          "foreach_cell.policy", 3,

          // Politique 1 : Kokkos::RangePolicy
          [&]() {
            std::cout << "Hello 1\n";
            Kokkos::parallel_for(
                kernel_name + "_range", Kokkos::RangePolicy<>(0, total_cells),
                KOKKOS_LAMBDA(uint32_t index) {
                  uint32_t iOct = index / cells_per_block;
                  uint32_t local_idx = index % cells_per_block;

                  uint32_t k = local_idx / (bx * by);
                  uint32_t j = (local_idx - k * bx * by) / bx;
                  uint32_t i = local_idx - j * bx - k * bx * by;

                  CellIndex iCell = {{iOct, false}, i, j, k, bx, by, bz};
                  f(iCell, index);
                });
          },

          // Politique 2 : Kokkos::MDRangePolicy
          [&]() {
            std::cout << "Hello 2\n";
            Kokkos::parallel_for(
                kernel_name + "_mdrange",
                Kokkos::MDRangePolicy<Kokkos::Rank<2>>(
                    {0, 0}, {num_octants, cells_per_block}),
                KOKKOS_LAMBDA(uint32_t oct, uint32_t index) {
                  uint32_t k = index / (bx * by);
                  uint32_t j = (index - k * bx * by) / bx;
                  uint32_t i = index - j * bx - k * bx * by;

                  CellIndex iCell = {{oct, false}, i, j, k, bx, by, bz};
                  f(iCell, oct * cells_per_block + index);
                });
          },

          // Politique 3 : Kokkos::TeamPolicy
          [&]() {
            std::cout << "Hello 3\n";
            using team_policy = Kokkos::TeamPolicy<>;
            using member_type = team_policy::member_type;

            const int team_size = 64;
            Kokkos::parallel_for(
                kernel_name + "_team", team_policy(num_octants, team_size),
                KOKKOS_LAMBDA(const member_type& team) {
                  const uint32_t iOct = team.league_rank();
                  Kokkos::parallel_for(
                      Kokkos::TeamThreadRange(team, cells_per_block),
                      [&](uint32_t index) {
                        uint32_t k = index / (bx * by);
                        uint32_t j = (index - k * bx * by) / bx;
                        uint32_t i = index - j * bx - k * bx * by;

                        CellIndex iCell = {{iOct, false}, i, j, k, bx, by, bz};
                        f(iCell, iOct * cells_per_block + index);
                      });
                });
          });
    }
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
    const uint32_t num_cells = shape.bx * shape.by * shape.bz * 20;

    Kokkos::View<int*> result("cell_sums", num_cells);

    // Appliquer une fonction simple à chaque cellule : i + j + k
    exec.foreach_cell(
        "foreach_demo", shape,
        KOKKOS_LAMBDA(CellIndex iCell, uint32_t flat_index) {
          result(flat_index) = iCell.i + iCell.j + iCell.k;
        });

    auto result_host =
        Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), result);

    std::cout
        << "Premiers résultats de i + j + k pour les premières cellules:\n";
    for (int i = 0; i < 10 && i < num_cells; ++i) {
      std::cout << "Cell " << i << " -> sum = " << result_host(i) << "\n";
    }

    Kokkos::fence();
  }
  Kokkos::finalize();
  return 0;
}

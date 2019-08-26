#ifndef LINKINGHEADER_C_H_
#define LINKINGHEADER_C_H_

extern "C"
{
	#include <vexs_io_functions.h>
}
#include <vexs_data_types_C.h>

#include <vector>

int gpu_launch( std::vector<float_type> energy_grids_flat,
				std::vector<float_type> xs_values_flat,
				std::vector<index_type> nuclide_positions,
				std::vector<index_type> material_positions,
				std::vector<index_type> material_sizes,
				std::vector<index_type> nuclide_ids,
				std::vector<index_type> nuclide_sizes,
				std::vector<float_type> probabilities,
				numerical_type total_materials,
				std::vector<float_type> concentrations,
				std::vector<index_type> hash_grid,
				numerical_type num_hash_bins,
				float_type du,
				float_type ln_energy_min,
				numerical_type num_lookups,
				numerical_type gpu_blocks,
				numerical_type gpu_threads);

#endif

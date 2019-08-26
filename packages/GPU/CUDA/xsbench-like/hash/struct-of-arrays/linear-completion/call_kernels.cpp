/*
This kernel is used to call whatever kernels the user has put into the kernels/ subfolder; make sure that the Makefile is configured
properly.
*/
//extern "C" {
#include "kernels/linkingheader_C.h"
//}

#include <vexs_header.h>

void call_kernels(problem_data problem){

	numerical_type num_hash_bins =  problem.hash_bins;

	problem.data1D_SOA.create_energy_grids_flat(problem); 
	problem.data1D_SOA.create_xs_values_flat(problem);
	problem.data1D_SOA.create_nuclide_positions(problem); 
	problem.data1D_SOA.create_material_positions(problem);
	problem.data1D_SOA.create_material_sizes(problem);
	problem.data1D_SOA.create_nuclide_ids(problem);
	problem.data1D_SOA.create_nuclide_sizes(problem);
	problem.data1D_SOA.create_probabilities(problem); //Create probabilities vector
	problem.data1D_SOA.create_concentrations(problem); //Create problem vector
	problem.data1D_SOA.create_hash_grid(problem, num_hash_bins); //Create hash grid

	gpu_launch(  problem.data1D_SOA.energy_grids_flat,
				 problem.data1D_SOA.xs_values_flat,
				 problem.data1D_SOA.nuclide_positions,
				 problem.data1D_SOA.material_positions,
				 problem.data1D_SOA.material_sizes,
				 problem.data1D_SOA.nuclide_ids,
				 problem.data1D_SOA.nuclide_sizes,
				 problem.data1D_SOA.probabilities,
				 problem.common_data.total_materials,
				 problem.data1D_SOA.concentrations,
				 problem.data1D_SOA.hash_grid,
				 num_hash_bins,
				 problem.data1D_SOA.du,
				 problem.data1D_SOA.ln_energy_min,
				 problem.num_lookups,
				 problem.gpu_blocks,
				 problem.gpu_threads);
}

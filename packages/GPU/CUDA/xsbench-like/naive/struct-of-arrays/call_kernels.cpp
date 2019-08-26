/*
This kernel is used to call whatever kernels the user has put into the kernels/ subfolder; make sure that the Makefile is configured
properly.
*/
#include "kernels/linkingheader_C.h"

#include <vexs_header.h>

void call_kernels(problem_data problem){

    problem.data1D_SOA.create_energy_grids_flat(problem); //Create 1D vector of all energy values
	problem.data1D_SOA.create_xs_values_flat(problem); //Create 1D vector of all xs values
	problem.data1D_SOA.create_nuclide_positions(problem); //Create vector indicating where nuclides are in the grid
	problem.data1D_SOA.create_material_positions(problem); //Create materials position map vector
	problem.data1D_SOA.create_material_sizes(problem); //Create materials position map vector
	problem.data1D_SOA.create_nuclide_ids(problem); //Create materials vector
	problem.data1D_SOA.create_nuclide_sizes(problem);
	problem.data1D_SOA.create_probabilities(problem); //Create probabilities vector
	problem.data1D_SOA.create_concentrations(problem); //Create problem vector

	gpu_launch( problem.data1D_SOA.energy_grids_flat,
				problem.data1D_SOA.xs_values_flat,
				problem.data1D_SOA.nuclide_positions,
				problem.data1D_SOA.material_positions,
				problem.data1D_SOA.material_sizes,
				problem.data1D_SOA.nuclide_ids,
				problem.data1D_SOA.nuclide_sizes,
				problem.data1D_SOA.probabilities,
				problem.common_data.total_materials,
				problem.data1D_SOA.concentrations,
				problem.num_lookups,
				problem.gpu_blocks,
				problem.gpu_threads);
}

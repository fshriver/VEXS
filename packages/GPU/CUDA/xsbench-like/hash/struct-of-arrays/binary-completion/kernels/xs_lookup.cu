/*
Main GPU lookup kernel; cannot be inlined, requires its own separate object files since it is __global__.
*/
#include "gpu_header.h"

__global__ void xs_lookup(float_type * energy_grids_flat,
						  float_type * xs_values_flat,
						  index_type * nuclide_positions,
						  index_type * material_positions,
						  index_type * material_sizes,
						  index_type * nuclide_ids,
						  index_type * nuclide_sizes,
						  float_type * probabilities,
						  numerical_type total_materials,
						  float_type * concentrations,
						  index_type * hash_grid,
						  numerical_type num_hash_bins,
						  float_type du,
						  float_type ln_energy_min,
						  numerical_type num_lookups,
						  float_type * results_array)
{
	index_type particle_mat;
	numerical_type n;
	numerical_type num_nuclides;
	float_type particle_energy;

	/*
	Generate an initial seed based on the threadid and blockid; each thread is guaranteed to generate its own unique sequence
	of random numbers.
	*/
	unsigned long seed = (threadIdx.x + (blockIdx.x * blockDim.x) + 1 )*19 + 17;
	rn(&seed);

	for (n = 0; n < num_lookups; n++)
	{
		/*
		Pick a random material to perform the lookup on.
		*/
		particle_mat = pick_mat( &seed, probabilities, total_materials);

		/*
		Pick a random energy for the particle
		*/
		#ifdef detailed
		particle_energy = rn(&seed) * 19.999e6;
		particle_energy = particle_energy * rn(&seed);
		particle_energy = particle_energy * rn(&seed);
		particle_energy = particle_energy * rn(&seed);
		particle_energy = particle_energy * rn(&seed);
		particle_energy = particle_energy * rn(&seed);
		particle_energy = particle_energy + 20;
		#endif
		#ifdef approximate
		particle_energy = rn(&seed) * 0.98 + 0.01;
		#endif
		/*
		The number of nuclides in the material is also located in the materials_position_map array.
		*/
		num_nuclides = material_sizes[ particle_mat ];

		/*
		Retrieve the actual location of the material in the materials array from the materials_position_map array.
		*/
		particle_mat = material_positions[ particle_mat ];

		calculate_macro_xs( energy_grids_flat,
							xs_values_flat,
							nuclide_positions,
							nuclide_ids,
							nuclide_sizes,
							concentrations,
							num_nuclides,
							hash_grid,
							num_hash_bins,
							du,
							ln_energy_min,
							particle_mat,
							particle_energy,
							results_array);
	}
}







/*
Calculates the macroscopic xs of the material (sum of each nuclide's microscopic cross section multiplied by its concentration
in the material).
*/
#ifndef __CALCULATE_MACRO_XS_I_CUH__
#define __CALCULATE_MACRO_XS_I_CUH__

__device__ inline void calculate_macro_xs( float_type * energy_grids_flat,
									float_type * xs_values_flat,
									index_type * nuclide_positions,
									index_type * nuclide_ids,
									index_type * nuclide_sizes,
									float_type * concentrations,
									numerical_type num_nuclides,
									index_type * hash_grid,
									numerical_type num_hash_bins,
									float_type du,
									float_type ln_energy_min,
									numerical_type particle_mat,
									float_type particle_energy,
									float_type * results_array)
{
	numerical_type i,k;
	index_type particle_nuclide;
	index_type nuclide_grid_location;
	index_type num_gridpoints;
	float_type conc;
	index_type idx = -1;
	volatile float_type macro_xs_vector[5];
	volatile float_type micro_xs_vector[5];
	index_type lower_grid_index = 0;
	index_type upper_grid_index = 0;

	/*
	Clear macroscopic xs vector.
	*/
	for (k = 0; k < 5; k++)
	{
		macro_xs_vector[k] = 0.0;
	}

	/*
	Calculate the coarse energy bin in the hash grid that our energy is in.
	*/
	idx = floor( du * ( log2(particle_energy) - ln_energy_min ) );

	/*
	Loop through the nuclides that are in the material; calculate the microscopic cross section of each nuclide, multiply it by
	that nuclide's concentration in the material, and add it to macroscopic xs array.
	*/
	for (i = 0; i < num_nuclides; i++)
	{
		particle_nuclide = nuclide_ids[ particle_mat + i];
		nuclide_grid_location = nuclide_positions[particle_nuclide];
		num_gridpoints = nuclide_sizes[ particle_mat + i];
		conc = concentrations[ particle_mat + i];
		lower_grid_index = hash_grid[ particle_nuclide * (num_hash_bins + 1) + idx ];
		upper_grid_index = hash_grid[ particle_nuclide * (num_hash_bins + 1) + idx + 1 ];
		upper_grid_index += 1 + ( upper_grid_index != (num_gridpoints - 1) );
		calculate_micro_xs( energy_grids_flat,
							xs_values_flat,
							nuclide_grid_location,
							num_gridpoints,
							particle_energy,
							lower_grid_index,
							upper_grid_index,
							micro_xs_vector);
		for ( k = 0; k < 5; k++ )
		{
			macro_xs_vector[k] += micro_xs_vector[k]*conc;
		}
	}
	
	for (k = 0; k < 5; k++)
	{
		results_array[threadIdx.x + (blockDim.x * blockIdx.x)] += macro_xs_vector[k];
	}
	
}
#endif

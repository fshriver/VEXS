/*
Declarations of inlined functions, so we don't get compile errors in our implementation files
*/

__device__ inline void calculate_macro_xs( float_type * energy_grids_flat,
										float_type * xs_values_flat,
										index_type * nuclide_positions,
										index_type * nuclide_ids,
										index_type * nuclide_sizes,
										float_type * concentrations,
										numerical_type num_nuclides,
										index_type particle_mat,
										float_type particle_energy,
										float_type * results_array);

__device__ inline void calculate_micro_xs( float_type * energy_grids_flat,
										float_type * xs_values_flat,
										index_type nuclide_grid_location,
										index_type num_gridpoints,
										float_type particle_energy,
										volatile float_type * micro_xs_vector);

__device__ inline numerical_type pick_mat( unsigned long * seed,
										float_type * probabilities,
										numerical_type total_materials);

__device__ inline float_type rn(unsigned long * seed);

__device__ inline index_type binary_search(index_type lower,
										index_type upper,
										float_type energy,
										float_type * energy_grid);
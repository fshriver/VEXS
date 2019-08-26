/*
Calculates the microscopic xs vector of a nuclide in a material, based on particle energy.
*/
#ifndef __CALCULATE_MICRO_XS_I_CUH__
#define __CALCULATE_MICRO_XS_I_CUH__

__device__ inline void calculate_micro_xs( float_type * energy_grids_flat,
									float_type * xs_values_flat,
									index_type nuclide_grid_location,
									index_type num_gridpoints,
									float_type particle_energy,
									index_type lower_grid_index,
									index_type upper_grid_index,
									volatile float_type * micro_xs_vector)
{
	index_type final_index;
	float_type f;
	float_type * energy_grid;
	float_type * xs_grid;

	energy_grid = &energy_grids_flat[nuclide_grid_location];
	xs_grid = &xs_values_flat[ nuclide_grid_location * 5];

	final_index = binary_search(lower_grid_index,
								upper_grid_index,
								particle_energy,
								energy_grid);

	//printf("lower: %f, energy: %f, upper: %f\n", nuclide_grids[particle_nuc].energy_grid[final_index], particle_energy, nuclide_grids[particle_nuc].energy_grid[final_index + 1]);
	/*
	if ( ( ( energy_grid[final_index] > particle_energy) || ( energy_grid[final_index + 1] < particle_energy ) ) && final_index != 0 )
	{
		printf("Computed an index that was not correct!\n");
		printf("Energy: %f\n", particle_energy);
		printf("Lower hash index: %d, upper hash index: %d\n", lower_grid_index, upper_grid_index + 1 );
		printf("Corresponding to energies: %f, and %f\n", energy_grid[lower_grid_index], energy_grid[upper_grid_index]);
		printf("Final computed lower index: %d, final computed upper index: %d\n", final_index, final_index + 1);
		printf("Corresponding to energies: %f, and %f\n", energy_grid[final_index], energy_grid[final_index + 1]);
		asm("trap;");
	}
	*/

	//xs = &nuclide_grids[particle_nuc].xs_values[ (final_index)*5 ];

	f = (energy_grid[final_index + 1] - particle_energy)/
	( energy_grid[final_index + 1] -
	energy_grid[final_index] );

	/*
	Total XS
	*/
	micro_xs_vector[0] = xs_grid[ final_index + 1 ] -
	f *( xs_grid[ final_index + 1 ] - 
		xs_grid[ final_index ] );

	/*
	Elastic XS
	*/
	micro_xs_vector[1] = xs_grid[ num_gridpoints + final_index + 1 ] -
	f *( xs_grid[ num_gridpoints + final_index + 1 ] - 
		xs_grid[ num_gridpoints + final_index ] );

	/*
	Absorbtion XS
	*/
	micro_xs_vector[2] = xs_grid[ (num_gridpoints * 2) + final_index + 1 ] -
	f *( xs_grid[ (num_gridpoints * 2) + final_index + 1 ] - 
		xs_grid[ (num_gridpoints * 2) + final_index ] );

	/*
	Fission XS
	*/
	micro_xs_vector[3] = xs_grid[ (num_gridpoints * 3) + final_index + 1 ] -
	f *( xs_grid[ (num_gridpoints * 3) + final_index + 1 ] - 
		xs_grid[ (num_gridpoints * 3) + final_index ] );

	/*
	Nu Fission XS
	*/
	micro_xs_vector[4] = xs_grid[ (num_gridpoints * 4) + final_index + 1 ] -
	f *( xs_grid[ (num_gridpoints * 4) + final_index + 1 ] - 
		xs_grid[ (num_gridpoints * 4) + final_index ] );

}
#endif





/* 

Function definitions for the 1D Struct-of-Arrays data layout.

*/
#include "vexs_header.h"

/*
Create vector of probabilities corresponding to material selection probability
*/
void data1DSOA::create_probabilities(problem_data problem)
{
	cout << "Creating 1D vector of material probabilities..." << endl;
	for (auto material_probability:problem.probability_map){
		probabilities.push_back(material_probability);
	}
}

/*
 * Create a vector indicating where each material id is in the nuclide_ids and nuclide_sizes arrays.
*/
void data1DSOA::create_material_positions(problem_data problem)
{
	cout << "Creating 1D vector mapping material indices to a flat array..." << endl;

	index_type position_counter = 0;
	for (auto material:problem.material_map)
	{
		material_positions.push_back(position_counter);
		position_counter = position_counter + material.size();
	}
}

/*
 * Create a vector indicating the size of each material in nuclides.
*/
void data1DSOA::create_material_sizes(problem_data problem)
{
	cout << "Creating 1D vector mapping material sizes to a flat array..." << endl;

	for (auto material:problem.material_map)
	{
		material_sizes.push_back(material.size());
	}
}

/*
 * Create a vector of the nuclide ids present in each material.
*/
void data1DSOA::create_nuclide_ids(problem_data problem)
{
	cout << "Creating nuclide ids..." << endl;

	for (auto material:problem.material_map)
	{
		for (auto nuclide:material)
		{
			nuclide_ids.push_back(nuclide);
		}
	}
}

/*
 * Create a vector of the size of each nuclide in each material
*/
void data1DSOA::create_nuclide_sizes(problem_data problem)
{
	cout << "Creating 1D vector mapping nuclides to their position in the nuclides grid..." << endl;

	vector<index_type> pos_vector;
	index_type pos_counter = 0;
	for (index_type nuclide = 0; nuclide < problem.nuclides.size(); nuclide++)
	{
		pos_vector.push_back( (*problem.nuclides[nuclide]).size() );
	}

	for (auto material:problem.material_map)
	{
		for (auto nuclide:material)
		{
			nuclide_sizes.push_back(pos_vector[nuclide]);
		}
	}
}

/*
Transform our 2D concentration map to a 1D representation.
*/
void data1DSOA::create_concentrations(problem_data problem)
{
	for (auto material:problem.concentrations_map)
	{
		for (auto nuclide:material)
		{
			concentrations.push_back(nuclide);
		}
	}
}

/*
Place all energy points in a single 1D vector.
*/
void data1DSOA::create_energy_grids_flat(problem_data problem)
{
	cout << "Creating 1D vector of energy values..." << endl;

	for (auto nuclide:problem.nuclides)
	{
		for ( auto point:*nuclide)
		{
			energy_grids_flat.push_back(point.energy);
		}
	}
}

/*
Places all xs values into a single 1D vector.
*/
void data1DSOA::create_xs_values_flat(problem_data problem)
{
	cout << "Creating 1D vector of xs values..." << endl;

	for (auto nuclide:problem.nuclides)
	{
		for (auto point:*nuclide)
		{
			xs_values_flat.push_back(point.total_xs);
		}
		for (auto point:*nuclide)
		{
			xs_values_flat.push_back(point.elastic_xs);
		}
		for (auto point:*nuclide)
		{
			xs_values_flat.push_back(point.absorption_xs);
		}
		for (auto point:*nuclide)
		{
			xs_values_flat.push_back(point.fission_xs);
		}
		for (auto point:*nuclide)
		{
			xs_values_flat.push_back(point.nu_fission_xs);
		}
	}
}

/*
Create vector mapping each nuclide id to its position in the actual flattened energy_grids_flat vector.
*/
void data1DSOA::create_nuclide_positions(problem_data problem)
{
	cout << "Creating 1D nuclide positions vector..." << endl;

	index_type pos_counter = 0;

	for (auto nuclide:problem.nuclides)
	{
		nuclide_positions.push_back(pos_counter);
		pos_counter = pos_counter + (*nuclide).size();
	}
}

/*
Creates a hash grid used to accelerate cross section lookup. This method is largely recycled from that used in the 2D
Array-of-Structs method, however the hash_grid is converted to a vector.
*/
void data1DSOA::create_hash_grid(problem_data problem, numerical_type number_of_bins)
{
	cout << "Creating 1D hash grid..." << endl;

	index_type e, i;

	long double energy_min = problem.common_data.energy_min;
	long double energy_max = problem.common_data.energy_max;

	/*
	Constant kernels based on a hashed search need a precomputed logarithm for computation
	*/
	ln_energy_min = log2(energy_min);

	/*
	Create temporary vector to hold our lethargy grid values
	*/
	vector<long double> lethargy_grid(number_of_bins);

	/*
	Form bins in lethargy space, ln(E_max/E_min) and calculate the corresponding energy values for each bin in the linear energy space
	*/
	numerical_type counter = 0;
	for (auto& point:lethargy_grid){
		point = ( log2( (long double)( (energy_max) /energy_min ) )/( (long double)number_of_bins ) ) * ( number_of_bins - counter);
		point = (long double)energy_max * exp2(-point);
		counter = counter + 1;
	}

	/*
	Compute constant we use in every hash grid search
	*/
	du = (float_type)( (float_type)number_of_bins/( log2(energy_max) - log2(energy_min) ) );

	/*
	Allocate hash grid, (number of bins + 1)*(number of nuclides)
	*/
	numerical_type n_nuclides = problem.nuclides.size();
	index_type ** hash_array = (index_type **)malloc( (number_of_bins + 1) * sizeof(index_type *));
	for (i = 0; i < (number_of_bins + 1); i++ ){
		hash_array[i] = (index_type *)calloc( problem.nuclides.size(), sizeof(index_type) );
	}

	/*
	Assign index values to hash grid. 
	*/
	for (i = 0; i < problem.nuclides.size(); i++)
	{
		index_type index = 1;
		index_type e = 0;
		index_type last_point = (*problem.nuclides[i]).size() - 2;
		while( index != lethargy_grid.size() + 1)
		{
			if( (*problem.nuclides[i])[e].energy > lethargy_grid[index] )
			{
				if (e == 0)
				{
					hash_array[index][i] = 0;
				}
				else
				{
					hash_array[index][i] = e - 1;
				}
				index = index + 1;
			} else if (e == last_point)
			{ //This part is specifically meant to deal with hash grid bins that are outside the max energy of the nuclide
				hash_array[index][i] = last_point;
				index = index + 1;
			} else
			{
				e = e + 1;
			}
		}
		hash_array[number_of_bins][i] = last_point;
	}

	
	for ( i = 0; i < problem.nuclides.size(); i++ )
	{
		for ( e = 0; e < (number_of_bins + 1); e ++ )
		{
			hash_grid.push_back( hash_array[e][i] );
		}
	}
	
	/*
 	Functionality for if you ever want to do it the other way instead
	*/
	/*
	for ( e = 0; e < (number_of_bins + 1); e ++ )
	{
		for ( i = 0; i < problem.nuclides.size(); i++ )
		{
			hash_grid.push_back( hash_array[e][i] );
		}
	}
	*/
}


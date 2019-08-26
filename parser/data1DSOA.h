/*

Class definitions for the 1D Struct-of-Arrays data layout.

*/
extern "C"
{
	#include "vexs_data_types_C.h"
}

class problem_data;

class data1DSOA{
public:
	/*
	Naive (binary) search
	*/
	vector<float_type> probabilities;
	vector<index_type> material_positions;
	vector<index_type> material_sizes;
	vector<index_type> nuclide_ids;
	vector<index_type> nuclide_sizes;
	vector<float_type> concentrations;
	vector<float_type> energy_grids_flat;
	vector<float_type> xs_values_flat;
	vector<index_type> nuclide_positions;

	/*
	Hash-grid search
	*/
	vector<index_type> hash_grid;
	float_type du;
	float_type ln_energy_min;

	void create_probabilities(problem_data problem);
	void create_material_positions(problem_data problem);
	void create_material_sizes(problem_data problem);
	void create_nuclide_ids(problem_data problem);
	void create_nuclide_sizes(problem_data problem);
	void create_concentrations(problem_data problem);
	void create_energy_grids_flat(problem_data problem);
	void create_xs_values_flat(problem_data problem);
	void create_nuclide_positions(problem_data problem);

	/*
	Hash grid search
	*/
	void create_hash_grid(problem_data problem, numerical_type number_of_bins);

};





/*

Definition for function to calculate common data used throughout different kernels; data is generally used and easy to compute.

*/
#include "vexs_header.h"
void commondata::calculate_common_data(problem_data problem){
	cout << "\n[Preparing common data]\n" << endl;
	numerical_type material, nuclide; //Needed declarations of variables we'll be using as indices

	/*
	Calculates the total number of energy points from all nuclide grids
	*/
	total_number_of_points = 0;
	for(auto nuclide:problem.nuclides){
		total_number_of_points = total_number_of_points + (*nuclide).size();
	}

	/*
	Calculates the total number of nuclides in the problem
	*/
	total_nuclides = problem.nuclides.size();

	/*
	Calculates the total number of materials in the problem
	*/
	total_materials = problem.material_map.size();

	/*
	Calculates the total number of nuclides used across all materials in the problem (this means nuclides that are present in more
	than one material are counted multiple times)
	*/
	total_nuclides_in_all_materials = 0;
	for (auto material:problem.material_map){
		total_nuclides_in_all_materials = total_nuclides_in_all_materials + material.size();
	}

	/*
	Produces a 1D array of ints, with each element indicating how many nuclides are in that specific material
	*/
	//Allocate array, initially set to all zeros.
	num_nuclides = (numerical_type *)calloc(problem.material_map.size() ,sizeof(int));

	//Loop through material_map, setting the array equal to the number of elements in that vector of material_map
	for (material = 0; material < problem.material_map.size(); material++){
		num_nuclides[material] = problem.material_map.at(material).size();
	}

	/*
	Creates a 1D array of ints holding the number of gridpoints in each nuclide grid
	*/
	num_gridpoints = (index_type *)calloc(problem.nuclides.size(), sizeof(long int));

	for(nuclide = 0; nuclide < problem.nuclides.size(); nuclide++){
		num_gridpoints[nuclide] =  (*problem.nuclides.at(nuclide)).size();
	}

	/*
	Produces a 2D jagged array of ints, with each int representing the index in the nuclides array 
	where a nuclide in the material can be found
	*/
	material_map = (numerical_type **)malloc(problem.material_map.size()*sizeof(int *));

	for (material = 0; material < problem.material_map.size(); material++){
		material_map[material] = (numerical_type *)calloc(problem.material_map.at(material).size(), sizeof(int));
		for (nuclide = 0; nuclide < problem.material_map.at(material).size(); nuclide++){
			material_map[material][nuclide] = problem.material_map.at(material).at(nuclide);
		}
	}

	/*
	Creates a 1D array of double values, which indicates the "probability" of a given material being selected 
	for cross section lookup.
	*/
	probability_map = (double *)calloc(problem.material_map.size(), sizeof(double));

	for(material = 0; material < problem.material_map.size(); material++){
		probability_map[material] = problem.probability_map.at(material);
	}

	/*
	Produces a 2D jagged array of doubles, with each double representing concentration of that nuclide in that material
	*/
	concentrations_map = (double **)malloc(problem.material_map.size()*sizeof(double *));

	for (material = 0; material < problem.material_map.size(); material++){
		concentrations_map[material] = (double *)calloc(problem.material_map.at(material).size(), sizeof(double));
		for (nuclide = 0; nuclide < problem.material_map.at(material).size(); nuclide++){
			concentrations_map[material][nuclide] = problem.concentrations_map.at(material).at(nuclide);
		}
	}

	/*
	Calculate the minimum energy across all nuclide grids.
	*/
	energy_min = (*problem.nuclides[0])[0].energy;
	for (auto nuclide:problem.nuclides)
	{
		if ( (*nuclide)[0].energy < energy_min)
		{
			energy_min = (*nuclide)[0].energy;
		}
	}
	if(energy_min == 0)
	{
		cout << "\n[ERROR]: There is an energy point in library that is equal to zero!" << endl;
		cout << "This is not physically possible!\n" << endl;
		exit(0);
	}

	/*
	Calculate the maximum energy across all nuclide grids.
	*/
	energy_max = (*problem.nuclides[0])[ (*problem.nuclides[0]).size() - 1 ].energy;
	for (auto nuclide:problem.nuclides)
	{
		if ( (*nuclide)[ (*nuclide).size() - 1 ].energy > energy_max )
		{
			energy_max = (*nuclide)[ (*nuclide).size() - 1 ].energy;
		}
	}
}






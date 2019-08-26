/*

Controls the assigning of randomized data to the nuclide grid structs and the material map.

*/
#include "vexs_header.h"

void assign_random_data(problem_data &problem){

	cout << "\n[Assigning random data]" << endl;

	unsigned long seed = 5; // TEMPORARY SEED

	//Kick the RNG off, since it initially gives a very low value
	random_number(seed);

	//If we have read in data from binary file, we don't need to assign values to nuclide grids, we assume they've been assigned.
	//We do still need to assign random values for the material nuclide concentrations, however.
	if (problem.binary){
		assign_random_concentrations(problem, seed);
		return;
	}

	assign_random_xs(problem, seed);

	seed = 5;

	random_number(seed);

	assign_random_concentrations(problem, seed);

	//This isn't really the best place to put this, but since this is the last place where we can modify the problem class
	//I think this is best; this just calculates and sets small numbers that users commonly want to know that aren't set
	//anywhere else, such as the number of nuclides in all nuclide grids, etc.


}

/*

Assigns random values to the material map

*/
#include "vexs_header.h"

void assign_random_concentrations(problem_data &problem, unsigned long &seed){
	cout << "\nAssigning random concentrations to materials..." << endl;

	for (auto material:problem.material_map){
		problem.concentrations_map.push_back(vector <double>());
		for (auto nuclide:material){
			problem.concentrations_map.back().push_back( random_number(seed) );
		}
	}
}
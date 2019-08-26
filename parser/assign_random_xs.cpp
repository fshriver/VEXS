/*

Assigns randomized reaction channel xs data to all nuclide points

*/
#include "vexs_header.h"
void assign_random_xs(problem_data &problem, unsigned long &seed){

	cout << "\nAssigning random cross section values..." << endl;
	for (auto& nuclide:problem.nuclides) {
		for (auto& point:*nuclide){
			point.total_xs = random_number(seed);
			point.elastic_xs = random_number(seed);
			point.absorption_xs = random_number(seed);
			point.fission_xs = random_number(seed);
			point.nu_fission_xs = random_number(seed);
		}
	}
}
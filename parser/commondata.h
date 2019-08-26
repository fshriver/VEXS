/*

Class definition for data that is very commonly used for many kernels, and relatively easy to compute.

Note: We explicitly put them all as long double because again, these data types (even the jagged 2D arrays) should not be the 
massive-memory part of the problem, and should also exist in the host memory, so putting these all as very large data types shouldn't
be overtly dangerous.

*/
extern "C"
{
	#include "vexs_data_types_C.h"
}
class problem_data;

class commondata{
public:
	index_type total_number_of_points;
	numerical_type total_nuclides;
	numerical_type total_nuclides_in_all_materials;
	numerical_type total_materials;
	numerical_type * num_nuclides;
	index_type * num_gridpoints;
	numerical_type ** material_map;
	double *probability_map;
	double **concentrations_map;
	double energy_min;
	double energy_max;

	void calculate_common_data(problem_data problem);
};
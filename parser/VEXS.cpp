/*

Main body of VEXS, calls the data reading/preparation functions as well as launches the kernels.

*/
#include "vexs_header.h"

using namespace std;

int main(int ac, char* av[]){

	/*
	Print logo
	*/
	print_vexs_logo();

	/*
	Create problem
	*/
	problem_data problem;

	/*
	Read in command-line information
	*/
	parse_cli(problem, ac, av);

	/*
	Read in library data
	*/
	read_library(problem);

	/*
	Read in data form binary file (faster loading)
	*/
	if (problem.binary){
		read_binary_data(problem);
	}

	/*
	Assign random values for cross sections and material concentrations in our problem
	*/
	assign_random_data(problem);

	/*
	Calculate cheap and easy-to-compute values data structures that are often used by kernels
	*/
	problem.common_data.calculate_common_data(problem);

	/*
	Print a summary of our problem parameters; at this points we should NOT be modifying the internal representation of the data anymore
	*/
	print_problem_data(problem);

	/*
	Write to binary file if user indicated they wanted that.
	*/
	if (problem.write_binary){
		write_binary_data(problem);
	}

	/*
	Call the kernels that are located in whatever directory the user has compiled VEXS in.
	*/
	call_kernels(problem);

}





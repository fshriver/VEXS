/*

Definitions of functions found in data_structures.h

*/
#include "vexs_header.h"

//Default constructor for an isotope gridpoint
nuclide_gridpoint::nuclide_gridpoint(){
	energy = 0.0;
	total_xs = 0.0;
	elastic_xs = 0.0;
	absorption_xs = 0.0;
	fission_xs = 0.0;
	nu_fission_xs = 0.0;
}

//Default constructor for a problem
problem_data::problem_data(){
	library = "test"; //This needs to be changed to default, test is a very simple directory for framework testing only
	lower_energy = 1e-5;
	upper_energy = 20e6;
	binary = FALSE;
	write_binary = FALSE;
	num_lookups = 15000000;
	hash_bins = 16384;
	mpi_processes = 1;
	openmp_threads = 1;
	gpu_blocks = 1;
	gpu_threads = 1;
}





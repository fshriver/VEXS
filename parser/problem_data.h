/*

Contains the data structures for the VEXS parser; while all of the class variables are public, they are not meant to
be passed to the kernels themselves. All members are public for the sake of brevity in the code.

*/
#include "commondata.h" //Common data that is usually needed/computed
#include "data1DSOA.h" //1D Struct-of-Arrays version of data

class nuclide_gridpoint{
public:
	long double energy;
	long double total_xs;
	long double elastic_xs;
	long double absorption_xs;
	long double fission_xs;
	long double nu_fission_xs;

	nuclide_gridpoint();
};

//Overall problem data
class problem_data{
public:
	string library;
	double lower_energy;
	double upper_energy;
	bool binary; //Indicates if we are reading in from a binary file
	bool write_binary; //Indicates if we are writing out to a binary file
	long int num_lookups;
	long int hash_bins;
	int mpi_processes;
	int openmp_threads;
	int gpu_blocks;
	int gpu_threads;
	vector<int> material_list; //List of material id's that have been read in
	vector<long double> probability_map; //Number of lookups or some other indicator to help us build a probability of interaction for each material
	vector< vector<int> > material_map; //Mapping of what nuclides are in each material; first is filled with id's from file, then gets converted to indices
	vector< vector<double> > concentrations_map;//Map of nuclide concentrations in each material
	vector<int> nuclide_list; //List of all nuclide ids, sorted numerically from least to greatest
	vector< vector<nuclide_gridpoint> *> nuclides; //2D array of all nuclide grids (should be in the same order as that of nuclide_list)
	commondata common_data;
	data1DSOA data1D_SOA;


	problem_data();
};

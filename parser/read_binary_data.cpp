/*

Reads in binary data from a file in the library directory; binary file is assumed to already exist (when you clone the repo it
shouldn't be included, so you'll have to generate it yourself).

Adapted from code written by John Tramm, found at https://github.com/ANL-CESAR/XSBench

*/
#include "vexs_header.h"

void read_binary_data(problem_data &problem){

	cout << "\nReading from binary file..." << endl;

	//Check if binary file exists
	string path = problem.library + "/data.bin";
	struct stat filecheck;
	if( stat(path.c_str(), &filecheck) != 0 ){
		cout << "[ERROR]: Binary file " << path << " does not exist.\n";
		exit(-1);
	}

	int stat;
	FILE * fp = fopen(path.c_str(), "rb");
	// Read Nuclide Grid Data
	for ( auto& nuclide:problem.nuclides){
		for ( auto& point:*nuclide){
			fread(&point.energy, sizeof(long double), 1, fp);
			fread(&point.total_xs, sizeof(long double), 1, fp);
			fread(&point.elastic_xs, sizeof(long double), 1, fp);
			fread(&point.absorption_xs, sizeof(long double), 1, fp);
			fread(&point.fission_xs, sizeof(long double), 1, fp);
			fread(&point.nu_fission_xs, sizeof(long double), 1, fp);
		}
	}

	fclose(fp);

}
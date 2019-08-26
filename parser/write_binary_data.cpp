/*

Writes nuclide grid data and unionized energy grid points to binary file.

Adapted from XSBench binary output code written by John Tramm, can be found at https://github.com/ANL-CESAR/XSBench

*/

#include "vexs_header.h"

void write_binary_data(problem_data problem){

	cout << "\n[Writing to binary file]" << endl;

	//Set binary file path
	string path = problem.library + "/data.bin";

	FILE * fp = fopen(path.c_str(), "wb");

	// Dump Nuclide Grid Data
	for ( auto nuclide:problem.nuclides ){
		for ( auto point:*nuclide){
			fwrite(&point.energy, sizeof(long double), 1, fp);
			fwrite(&point.total_xs, sizeof(long double), 1, fp);
			fwrite(&point.elastic_xs, sizeof(long double), 1, fp);
			fwrite(&point.absorption_xs, sizeof(long double), 1, fp);
			fwrite(&point.fission_xs, sizeof(long double), 1, fp);
			fwrite(&point.nu_fission_xs, sizeof(long double), 1, fp);
		}
	}

	fclose(fp);

}

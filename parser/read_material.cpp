/*

Reads in material information from the material files.

*/
#include "vexs_header.h"
#include <fstream>
#include <boost/algorithm/string.hpp>

//using namespace boost;

void read_material(problem_data &problem, int material){

	//Check if material file exists
	string path = problem.library + "/mat" + to_string(material) + ".txt";
	struct stat filecheck;
	if( stat(path.c_str(), &filecheck) != 0 || !(S_ISREG(filecheck.st_mode)) ){
		cout << "[ERROR]: Material file " << path << " does not exist.\n";
		exit(-1);
	}

	/*
	We fetch the first line and read in the value as an integer, telling us how far down we have to read to get all 
	nuclide information for the material; we assume the user has been diligent enough to make sure this number is 
	accurate, as the number of nuclides in a material is very broadly important information for the simulation
	*/
	ifstream file(path);
	string token;
	string line;
	getline(file,line);
	token = line.substr( 0, line.find(" ") );
	int n_nuclides = stoi(token);
	token.clear();
	for (int i = 0; i < n_nuclides; i++){
		getline(file, line);
		token = line.substr( 0, line.find(" ") );
		problem.material_map.back().push_back( stoi( token ) );
		token.clear();
	}

}

/*

Reads in nuclide grid data from the nuclide grid files

*/
#include "vexs_header.h"
#include <boost/algorithm/string.hpp>

//using namespace boost;

void read_nuclide(problem_data &problem, int nuclide){

	//Check if nuclide file exists
	string path = problem.library + "/ng" + to_string(nuclide) + ".txt";
	struct stat filecheck;
	if( stat(path.c_str(), &filecheck) != 0 || !(S_ISREG(filecheck.st_mode)) ){
		cout << "[ERROR]: Nuclide grid file " << path << " does not exist.\n";
		exit(-1);
	}

	/*
	We fetch the first line and read in the value as an integer, telling us how far down we have to read to get all 
	energy point information for the nuclide; we assume the user has been diligent enough to make sure this number is 
	accurate, as the number of energy points in a nuclide grid is very broadly important information for the simulation
	*/
	ifstream file(path);
	string  token;
	string line;

	getline(file,line);
	token = line.substr( 0, line.find(" ") );
	//split(token, line, is_any_of(" "), token_compress_on);
	int n_points = stoi(token);
	token.clear();

	//If we are reading from a binary file, push back an empty vector because we'll be loading it in later
	if(problem.binary){
		nuclide_gridpoint point;
		for (int i = 0; i < n_points; i++){
			(*problem.nuclides.back()).push_back(point);
		}
		return;
	}

	for (int i = 0; i < n_points; i++){
		getline(file, line);
		token = line.substr( 0, line.find(" ") );
		nuclide_gridpoint point;
		point.energy = stold(token);
		(*problem.nuclides.back()).push_back(point);
		token.clear();
	}
}

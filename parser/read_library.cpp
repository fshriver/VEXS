/*

Reads in "library" to be used by VEXS; library in this case meaning the data directory containing nuclide energy
grid data as well as material data

*/
#include "vexs_header.h"
#include <unistd.h>
#include <string.h>
#include <boost/algorithm/string.hpp>

//using namespace boost;

void read_library(problem_data &problem){

	cout << "\n[Reading library]" << endl;
	struct stat dircheck;

	char cwd[1024];
	getcwd(cwd, sizeof(cwd));
	string currentdir = cwd;
	string token, vexsdir;
	size_t pos = 0;

	currentdir.erase(0,1);
	while ( ( pos = currentdir.find("/") ) != std::string::npos )
	{
		token = currentdir.substr( 0, pos );
		if ( token == "VEXS" )
		{
			vexsdir.append("/");
			vexsdir.append(token);
			break;
		}
		else
		{
			vexsdir.append("/");
			vexsdir.append(token);
		}
		currentdir.erase(0, pos + 1);
	}

	if ( vexsdir.find("VEXS") == std::string::npos )
	{
		cout << "Path does not contain any form of VEXS; we need this to know where the libraries are!" << endl;
	}

	problem.library = vexsdir + "/DATA/" + problem.library;

	//Check if library exists
	if( stat(problem.library.c_str(), &dircheck) != 0 || !(S_ISDIR(dircheck.st_mode)) ){
		cout << "[ERROR]: Library path " << problem.library <<" not found.\n";
		exit(-1);
	}

	//Print out library location information
	cout << "\nLibrary at: " << problem.library << "\n";

	cout << "\nReading metadata..." << endl;

	//If library exists, try to read metadata for it.
	read_metadata(problem);

	//Sort material list and remove all duplicates
	sort(problem.material_list.begin(), problem.material_list.end());
	auto material_iterator = unique(problem.material_list.begin(), problem.material_list.end());
	problem.material_list.resize(distance(problem.material_list.begin(), material_iterator));

	cout << "\nReading materials..." << endl;

	//Iterate through material_map vector, building vectors of the nuclides that each material contains
	for(auto material:problem.material_list){
		problem.material_map.push_back(vector<int>());
		read_material(problem, material);
	}

	//Create the nuclide list; loops over all elements of material_map and puts them into a single <int> vector
	for (auto material:problem.material_map){
		for (auto nuclide:material){
			problem.nuclide_list.push_back(nuclide);
		}
	}

	//Sort and remove duplicates in nuclide list, so we don't have extra nuclides assembled
	sort(problem.nuclide_list.begin(), problem.nuclide_list.end());
	auto nuclide_iterator = unique(problem.nuclide_list.begin(), problem.nuclide_list.end());
	problem.nuclide_list.resize(distance(problem.nuclide_list.begin(), nuclide_iterator));

	cout << "\nReading nuclide grids..." << endl;

	//Iterate through nuclides vector, building vectors of structs that represent each nuclide grid
	for (auto& nuclide:problem.nuclide_list){
		problem.nuclides.push_back(new vector<nuclide_gridpoint>(0));
		read_nuclide(problem, nuclide);
	}

	//Iterate through each material in material map, finding where each nuclide in that material is in the nuclide grid and 
	//replacing the ZAIDs we read in from the input files with their actual place on the index
	for (auto& material:problem.material_map){
		for (auto& nuclide:material){
			int index = find(problem.nuclide_list.begin(), problem.nuclide_list.end(), nuclide) - problem.nuclide_list.begin();
			nuclide = index;
		}
	}

	//Calculate the probability of interacting with each material
	long double sum = 0.0;
	for(auto probability:problem.probability_map){
		sum = sum + probability;
	}
	long double new_probability = 0.0;
	for (auto& probability:problem.probability_map){
		new_probability = (long double)(probability/sum);
		probability = new_probability;
	}
}







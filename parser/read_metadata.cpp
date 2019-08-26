#include "vexs_header.h"
#include <boost/algorithm/string.hpp>

//using namespace boost;

void read_metadata(problem_data &problem){

	//Check if metadata file exists
	string path = problem.library + "/metadata.txt";
	struct stat filecheck;
	if( stat(path.c_str(), &filecheck) != 0 || !(S_ISREG(filecheck.st_mode)) ){
		cout << "[ERROR]: Metadata file " << path << " does not exist.";
		exit(-1);
	}

	ifstream file (path);
	string token;
	string line;
	while( getline(file, line) )
	{
		line.erase( 0, line.find(" ") + 1 );
		token = line.substr( 0, line.find(" "));
		problem.material_list.push_back( stoi( token ) );
		line.erase(0, line.find(" ") + 1);
		token = line.substr( 0, line.find(" "));
		problem.probability_map.push_back( stold( token ) );
	}
}

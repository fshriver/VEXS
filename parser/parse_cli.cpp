/*

Parses command line options; currently supported options are the directory where we want VEXS to look for simulation
data, and the type of structure we want VEXS to output for exposure to our kernel.

*/
#include "vexs_header.h"
#include <boost/program_options.hpp>

using namespace std;

void parse_cli(problem_data &problem, int argc, char* argv[])
{
	string argument;

	for ( int i = 1; i < argc; ++i )
	{
		argument = std::string(argv[i]);
		if ( argument == "-h" | argument == "--help" )
		{
			cout << "Program usage:" << endl;
			cout << "--help,-h : Outputs this help message." << endl;
			cout << "--input,-i : Directory under the DATA folder where VEXS will look for library data. Default path: DATA/test" << endl;
			cout << "--write-binary,-w : Write library data to binary file for faster reading in the future." << endl;
			cout << "--binary-b : Load library data from a binary file as opposed to reading from a text file." << endl;
			cout << "--lookups,-l : Total number of macroscopic cross-section lookups to perform." << endl;
			cout << "--hashbins,-h : Number of hash-bins to use in the hash-accelerated search." << endl;
			cout << "--openmpthreads : Number of OpenMP threads to use. Only applicable for OpenMP-specific kernels." << endl;
			cout << "--gpublocks : Number of blocks to use for GPU-specific kernels." << endl;
			cout << "--gputhreads: Number of threads per block to use for GPU-specific kernels." << endl;
			exit(0);
		}

		else if ( argument == "--input" | argument == "-i" )
		{
			if ( i + 1 < argc )
			{
				i++;
				problem.library = std::string(argv[i]);
			}
			else
			{
				cout << "[ERROR]: " << argument << " argument requires a value." << endl;
				exit(1);
			}
		}
		
		else if ( argument == "--write-binary" | argument == "-w" )
		{
			problem.write_binary = TRUE;
		}
		
		else if ( argument == "--binary" | argument == "-b" )
		{
			problem.binary = TRUE;
		}

		else if ( argument == "--lookups" | argument == "-l" )
		{
			if ( i + 1 < argc )
			{
				i++;
				problem.num_lookups = stoi( std::string(argv[i]) );
			}
			else
			{
				cout << "[ERROR]: " << argument << " argument requires a value." << endl;
				exit(1);
			}
		}
	
		else if ( argument == "--hashbins" | argument == "-h" )
		{
			if ( i + 1 < argc )
			{
				i++;
				problem.hash_bins = stoi( std::string(argv[i]) );
			}
			else
			{
				cout << "[ERROR]: " << argument << " argument requires a value." << endl;
				exit(1);
			}
		}

		else if ( argument == "--openmpthreads" )
		{
			if ( i + 1 < argc )
			{
				i++;
				problem.openmp_threads = stoi( std::string(argv[i]) );
			}
			else
			{
				cout << "[ERROR]: " << argument << " argument requires a value." << endl;
				exit(1);
			}
		}

		else if ( argument == "--gpublocks" )
		{
			if ( i + 1 < argc )
			{
				i++;
				problem.gpu_blocks = stoi( std::string(argv[i]) );
			}
			else
			{
				cout << "[ERROR]: " << argument << " argument requires a value." << endl;
				exit(1);
			}
		}

		else if ( argument == "--gputhreads" )
		{
			if ( i + 1 < argc )
			{
				i++;
				problem.gpu_threads = stoi( std::string(argv[i]) );
			}
			else
			{
				cout << "[ERROR]: " << argument << " argument requires a value." << endl;
				exit(1);
			}
		}
		
		else
		{
			cout << "[ERROR]: Passed an unrecognized argument " << argument << endl;
			exit(1);
		}
	}
}

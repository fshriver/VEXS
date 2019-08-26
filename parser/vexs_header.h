/*

VEXS header file; declares common functions and also defines the available data types

*/
#ifndef __VEXS_HEADER_H__
#define __VEXS_HEADER_H__
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <sys/stat.h>
#include <fstream>
#include <algorithm>
#include <cmath>

//Standard namespaces
using namespace std;

//VEXS data structures definitions
#include "problem_data.h"

//Common IO functions; we give them their own separate header file so we can expose them to CUDA kernels, which can actually use them
extern "C"
{
	#include "vexs_io_functions.h"
}

//VEXS exposed data structure definitions
//I don't think we need this anymore, let's comment it out and see what happens!
/*
#define IGAOS 1 //Individual grid, array of structs
#define IGFA 2 //Individual grid, flat array (one single large array)
#define IGFAI 3 //Individual grids, flat arrays, each nuclide has its xs values interleaved after the energy grid [DEFAULT]
#define UGAOS 4 //Unionized grid, array of structs
#define UGFA 5 //Unionized grid, flat array
#define ALL 6 //All above data types
*/

//Define flag values for reading in binary files
#define FALSE 0
#define TRUE 1

//Command-line parsing functions
void parse_cli(problem_data &problem, int ac, char* av[]);

//Library reading functions
void read_library(problem_data &problem);
void read_metadata(problem_data &problem);
void read_material(problem_data &problem, int material);
void read_nuclide(problem_data &problem, int nuclide);

//Random data assignment functions
void assign_random_data(problem_data &problem);
void assign_random_xs(problem_data &problem, unsigned long &seed);
void assign_random_concentrations(problem_data &problem, unsigned long &seed);

//Generation of random numbers
long double random_number(unsigned long &seed);

//VEXS functions to binary file I/O
void write_binary_data(problem_data problem);
void read_binary_data(problem_data &problem);

void print_problem_data(problem_data problem);
void call_kernels(problem_data problem);

/*
Just prints the VEXS logo
*/
void print_vexs_logo();
#endif







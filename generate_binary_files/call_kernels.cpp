/*
This function is used to call whatever kernels the user has put into the kernels/ subfolder; make sure that the Makefile is configured
properly.
*/

#include <vexs_header.h>

void call_kernels(problem_data problem)
{
	cout << "\nSuccessfully generated binary file for library located at: " << problem.library << endl;
}

/*

Prints a summary of the information read in and interpreted by VEXS; border_print and center_print shamelessly stolen from the
XSBench repo, found here: https://github.com/ANL-CESAR/XSBench

*/
#include "vexs_header.h"
#include <iomanip>
#include <string.h>

void center_print(const char *s, int width);
void border_print(void);

void print_problem_data(problem_data problem){
	border_print();
	center_print("PROBLEM SUMMARY", 79);
	border_print();

	//Print our the indices of each nuclide in the nuclide vector
	cout << "\nMap of nuclides to their indices in the nuclide array:\n" << endl;
	int index;
	for(index = 0; index < problem.nuclide_list.size(); index++){
		cout << "   " << "Nuclide " << problem.nuclide_list[index] << " --- index " << index << endl;
	}

	cout << "\nMap of materials to their indices in the materials array:\n" << endl;
	for(index = 0; index < problem.material_map.size(); index++){
		cout << "   " << "Material " << problem.material_list[index] << "   --- index " << index << endl;
	}

	//Print the nuclides that are in each material - the material_map has already been remapped to the indices of that nuclide
	//in the nuclide grid, as opposed to the ZAID's that it originally was filled with
	cout << "\nMap of nuclides that are in each material; numbers correspond to above indices:\n" << endl;
	for(index = 0; index < problem.material_map.size(); index++){
		cout << "   " << "Material " << index <<": ";
		for(auto nuclide:problem.material_map[index]){
			cout << nuclide << " ";
		}
		cout << endl;
	}

	//Print the concentrations each nuclide has in a given material
	cout << "\nMap of concentrations of nuclides for each material; corresponds to above material map and indices:\n" << endl;
	for(index = 0; index < problem.material_map.size(); index++){
		cout << "   " << "Material " << index <<": ";
		for(auto concentration:problem.concentrations_map[index]){
			cout << fixed << setprecision(3) << concentration << " ";
		}
		cout << endl;
	}

	//Print the "probability" of interacting with each material
	cout << "\n\"Probability\" of interaction for each material:\n" << endl;
	for(index = 0; index < problem.probability_map.size(); index++){
		cout << fixed << setprecision(6) << "   " << "Material " << problem.material_list[index] 
		<<": " << problem.probability_map[index] << endl;
	}


	//Print the total number of energy points in all nuclide grids
	/*
	int counter = 0;
	for(auto nuclide:problem.nuclides){
		counter = counter + (*nuclide).size();
	}
	*/
	cout << "\n   Total number of materials: " << problem.common_data.total_materials << endl;
	cout << "   Total number of nuclides: " << problem.common_data.total_nuclides << endl;
	cout << "   Total number of points in all nuclide energy grids: " << problem.common_data.total_number_of_points << endl;
	cout << endl;

	border_print();
	center_print("END PROBLEM SUMMARY", 79);
	border_print();

}





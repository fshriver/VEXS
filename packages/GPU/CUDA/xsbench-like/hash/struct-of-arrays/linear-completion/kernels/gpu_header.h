/*
Header containing declaration of our lookup kernel, as well as includes for our inlined functions.
*/

#ifndef __GPU_HEADER__
#define __GPU_HEADER__
/*
Include the header linking our gpu functions to the C++ part of VEXS.
*/
#include "linkingheader_C.h"

/*
Include Nvidia THRUST libraries used for easy data transfer.
*/
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/device_ptr.h>

/*
Include needed header files for time recording and file I/O
*/
#include <iostream>
#include <chrono>
#include <fstream>

/*
Include inlined device function declarations
*/
#include "inline_function_declarations.cuh"

/*
Include inlined device function definitions
*/
#include "calculate_macro_xs.i.cuh"
#include "calculate_micro_xs.i.cuh"
#include "linear_search.i.cuh"
#include "pick_mat.i.cuh"
#include "rn.i.cuh"

__global__ void xs_lookup(float_type * energy_grids_flat,
						float_type * xs_values_flat,
						index_type * nuclide_positions,
						index_type * material_positions,
						index_type * material_sizes,
						index_type * nuclide_ids,
						index_type * nuclide_sizes,
						float_type * probabilities,
						numerical_type total_materials,
						float_type * concentrations,
						index_type * hash_grid,
						numerical_type num_hash_bins,
						float_type du,
						float_type ln_energy_min,
						numerical_type num_lookups,
						float_type * results_array);

/*
Assertion to check whether a kernel launch/memory operation (using pure CUDA) was valid. If not, code != cudaSuccess and 
error is thrown telling the user what kind and on what line.
*/
#define gpu_error_check(ans) { gpu_throw_error((ans), __FILE__, __LINE__); }
inline void gpu_throw_error(cudaError_t code, const char *file, int line)
{
	if (code != cudaSuccess) 
	{
		fprintf(stderr,"\nGPU ERROR: %s at %s, line %d\n\nm", cudaGetErrorString(code), file, line);
		exit(code);
	}
}

#endif





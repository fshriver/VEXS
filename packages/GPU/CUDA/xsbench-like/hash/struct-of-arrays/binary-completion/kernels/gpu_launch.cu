/*
Launching function used to place the data on the device; uses THRUST libraries to allocate data on device, and then creates raw 
pointers to that data that are passed to the CUDA kernels that perform the actual lookups.
*/
#include "gpu_header.h"

using namespace std;

int gpu_launch(  vector<float_type> energy_grids_flat,
				 vector<float_type> xs_values_flat,
				 vector<index_type> nuclide_positions,
				 vector<index_type> material_positions,
				 vector<index_type> material_sizes,
				 vector<index_type> nuclide_ids,
				 vector<index_type> nuclide_sizes,
				 vector<float_type> probabilities,
				 numerical_type total_materials,
				 vector<float_type> concentrations,
				 vector<index_type> hash_grid,
				 numerical_type num_hash_bins,
				 float_type du,
				 float_type ln_energy_min,
				 numerical_type num_lookups,
				 numerical_type gpu_blocks,
				 numerical_type gpu_threads)
{
	ofstream timing;

	border_print();
	center_print("GPU Cross-Section Lookup Kernel, Hash-based search, Struct-of-Arrays",80);
	border_print();

	/*
	Calculate number of lookups each thread will have to perform.
	*/
	numerical_type num_lookups_per_thread = ceil( (float_type)num_lookups / ( (float_type) gpu_blocks * gpu_threads ) );

	cout << "Number of hash bins: " << num_hash_bins << endl;
	cout << "Number of blocks on GPU: " << gpu_blocks << endl;
	cout << "Number of threads per block on GPU: " << gpu_threads << endl;
	cout << "Number of lookups per thread: " << num_lookups_per_thread << endl;

	thrust::device_vector<float_type> DEVICE_energy_grids_flat = energy_grids_flat;
	thrust::device_vector<float_type> DEVICE_xs_values_flat = xs_values_flat;
	thrust::device_vector<index_type> DEVICE_nuclide_positions = nuclide_positions;
	thrust::device_vector<index_type> DEVICE_material_positions = material_positions;
	thrust::device_vector<index_type> DEVICE_material_sizes = material_sizes;
	thrust::device_vector<index_type> DEVICE_nuclide_ids = nuclide_ids;
	thrust::device_vector<index_type> DEVICE_nuclide_sizes = nuclide_sizes;
	thrust::device_vector<float_type> DEVICE_probabilities = probabilities;
	thrust::device_vector<float_type> DEVICE_concentrations = concentrations;
	thrust::device_vector<index_type> DEVICE_hash_grid = hash_grid;
	vector<float_type> results_array( gpu_blocks*gpu_threads, 0.0);
	thrust::device_vector<float_type> DEVICE_results_array = results_array;

	/*
	Create the raw pointers to pass to the GPU kernels (these are pointers to where the data actually is on the device).
	*/
	float_type * energy_grids_flat_device_ptr = thrust::raw_pointer_cast(&DEVICE_energy_grids_flat[0]);
	float_type * xs_values_flat_device_ptr = thrust::raw_pointer_cast(&DEVICE_xs_values_flat[0]);
	index_type * nuclide_positions_device_ptr = thrust::raw_pointer_cast(&DEVICE_nuclide_positions[0]);
	index_type * material_positions_device_ptr = thrust::raw_pointer_cast(&DEVICE_material_positions[0]);
	index_type * material_sizes_device_ptr = thrust::raw_pointer_cast(&DEVICE_material_sizes[0]);
	index_type * nuclide_ids_device_ptr = thrust::raw_pointer_cast(&DEVICE_nuclide_ids[0]);
	index_type * nuclide_sizes_device_ptr = thrust::raw_pointer_cast(&DEVICE_nuclide_sizes[0]);
	float_type * probabilities_device_ptr = thrust::raw_pointer_cast(&DEVICE_probabilities[0]);
	float_type * concentrations_device_ptr = thrust::raw_pointer_cast(&DEVICE_concentrations[0]);
	index_type * hash_grid_device_ptr = thrust::raw_pointer_cast(&DEVICE_hash_grid[0]);
	float_type * results_array_ptr = thrust::raw_pointer_cast(&DEVICE_results_array[0]);


	timing.open("timing.txt", std::fstream::app);
	auto start = std::chrono::high_resolution_clock::now();

	/*
	Call GPU kernel; this calls the CUDA kernel that will actually perform the cross section lookup.
	*/
	xs_lookup<<< gpu_blocks, gpu_threads >>>(energy_grids_flat_device_ptr,
											 xs_values_flat_device_ptr,
											 nuclide_positions_device_ptr,
											 material_positions_device_ptr,
											 material_sizes_device_ptr,
											 nuclide_ids_device_ptr,
											 nuclide_sizes_device_ptr,
											 probabilities_device_ptr,
											 total_materials,
											 concentrations_device_ptr,
											 hash_grid_device_ptr,
											 num_hash_bins,
											 du,
											 ln_energy_min,
											 num_lookups_per_thread,
											 results_array_ptr);
	gpu_error_check( cudaPeekAtLastError() );
	cudaDeviceSynchronize();
	
	/* 
	Stop clock
	*/
	//t = clock() - t;
	auto finish = std::chrono::high_resolution_clock::now();

	/*
	Calculate total time taken
	*/
	std::chrono::duration<float_type> elapsed_time = finish - start;

	/*
	Copy results back to host. This is so we know that the compiler won't accidentally optimize out our calculations.
	*/
	float_type * host_results_array =  (float_type *)malloc(gpu_threads * gpu_blocks * sizeof(float_type) );
	cudaMemcpy(host_results_array, results_array_ptr, gpu_threads * gpu_blocks * sizeof(float_type), cudaMemcpyDeviceToHost );

	
	cout << "Checksum is: " << endl;
	double checksum = 0.0;
	for (int i = 0; i < gpu_blocks * gpu_threads; i++){
		checksum += ( host_results_array[i] );
	}
	cout << checksum << endl;
	
	/*
	Figure out how many lookups we are actually doing
	*/
	numerical_type total_lookups = 0;
	if ( num_lookups < (gpu_threads * gpu_blocks) )
	{
		total_lookups = gpu_threads * gpu_blocks;
	}
	else
	{
		total_lookups = num_lookups;
	}

	cout << "Rate = " << std::scientific << total_lookups/( elapsed_time.count() ) << endl;

	/*
	Write out timing information
	*/
	timing << num_hash_bins << " " <<
	total_lookups << " " <<
	gpu_blocks << " " <<
	gpu_threads  << " "<<
	elapsed_time.count() << " " <<
	total_lookups / ( elapsed_time.count() ) << " " << 
	fixed << checksum << endl;
	timing.close();

	return 0;


}








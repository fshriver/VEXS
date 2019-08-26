/*
Performs a binary search on a nuclide grid; returns lower index.
*/
#ifndef __BINARY_SEARCH_I_CUH__
#define __BINARY_SEARCH_I_CUH__

__device__ inline index_type binary_search(index_type lower,
									index_type upper,
									float_type energy,
									float_type * energy_grid)
{
	index_type lower_limit = lower;
	index_type upper_limit = upper;
	index_type examination_point;
	index_type length = upper_limit - lower_limit;

	while (length > 1)
	{
		examination_point = lower_limit + (length / 2);
		if ( energy_grid[examination_point] > energy )
		{
			upper_limit = examination_point;
		}
		else
		{
			lower_limit = examination_point;
		}
		length = upper_limit - lower_limit;
	}

	return lower_limit;
}
#endif

#ifndef __LINEAR_SEARCH_I_CUH__
#define __LINEAR_SEARCH_I_CUH__

__device__ inline index_type linear_search(index_type lower,
							 index_type upper,
							 float_type energy,
							float_type * energy_grid)
{
	int i;

	//if ( (upper - lower) == 1 )
	//{
	//	return lower;
	//}

	for (i = 0; i <= (upper - lower); i++)
	{
		if ( energy_grid[lower + i] > energy)
		{
			return lower + i - 1;
		}
	}

	return upper;
}
#endif

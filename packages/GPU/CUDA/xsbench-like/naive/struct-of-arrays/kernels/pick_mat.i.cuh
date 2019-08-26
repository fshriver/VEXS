/*
Picks a random material according to a given probability distribution.
*/
#ifndef __PICK_MAT_I_CUH__
#define __PICK_MAT_I_CUH__

__device__ inline numerical_type pick_mat( unsigned long * seed,
										float_type * probabilities,
										numerical_type total_materials)
{
	float_type roll = rn(seed);

	numerical_type i, j;

	for ( i = 0; i < total_materials; i++)
	{
		float_type running = 0.0;
		for (j = i; j > 0; j--)
		{
			running += probabilities[j];
		}
		if (roll < running)
		{
			return i;
		}
	}

	return 0;
}
#endif
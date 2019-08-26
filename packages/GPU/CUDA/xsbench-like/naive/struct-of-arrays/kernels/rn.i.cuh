/*
Pick a random number
*/
#ifndef __RN_I_CUH__
#define __RN_I_CUH__
__device__ inline float_type rn(unsigned long * seed)
{
	float_type ret;
	unsigned long n1;
	unsigned long a = 16807;
	unsigned long m = 2147483647;
	n1 = ( a * (*seed) ) % m;
	*seed = n1;
	ret = (float_type) n1 / m;
	return ret;
}
#endif
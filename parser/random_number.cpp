/*

Algorithm for outputting random values; copied from John Tramm's code, found at https://github.com/ANL-CESAR/XSBench

*/
#include "vexs_header.h"

// Park & Miller Multiplicative Conguential Algorithm
// From "Numerical Recipes" Second Edition
long double random_number(unsigned long &seed){
	long double ret;
	unsigned long n1;
	unsigned long a = 16807;
	unsigned long m = 2147483647;
	n1 = ( a * (seed) ) % m;
	seed = n1;
	ret = (long double) n1 / m;
	return ret;
}
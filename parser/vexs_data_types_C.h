#ifndef __VEXS_DATA_TYPES_C_H__
#define __VEXS_DATA_TYPES_C_H__

#include <stdint.h>
/*
Defining data type sizes so we don't overallocate memory. This isn't really a problem for CPUs most of the time but
can be a problem for GPUs.
*/
typedef int32_t index_type;
typedef int32_t numerical_type;
typedef double float_type;

typedef struct{
	double energy;
	double total_xs;
	double elastic_xs;
	double absorption_xs;
	double fission_xs;
	double nu_fission_xs;
} NuclideGridPoint;

typedef struct{
	double energy;
	index_type * xs_ptrs;
} UnionizedGridPoint;

typedef struct{
	double * energy_grid;
	double * xs_values;
} NuclideGrid;

typedef struct{
	double * energy_grid;
	index_type ** xs_ptrs;
} UnionizedGrid;

typedef struct {
	double * energy_grid;
	index_type * xs_ptrs;
} FlatUnionizedGrid;


#endif

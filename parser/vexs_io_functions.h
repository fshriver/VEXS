/*
IO formatting functions that come in handy and are used in VEXS; feel free to reference these in your own code, as their 
definitions are in C (despite the file extension) and so if you wish to include them in a C kernel you can just include
as normal; if you wish to use them in a C++/CUDA kernel, you will need to wrap the include in an extern "C" block, like so:

extern "C" 
{
	#include <vexs_io_functions.h>
}

*/
#ifndef __VEXS_IO_FUNCTIONS_H__
#define __VEXS_IO_FUNCTIONS_H__
#include <stdio.h>
#include <string.h>

void border_print(void);
void center_print(const char *s, int width);
void fancy_int( long a );

#endif
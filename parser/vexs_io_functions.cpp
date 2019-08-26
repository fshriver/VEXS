/*

Generic output functions that are purely intended to display text; taken from code originall written by John Tramm at 
https://github.com/ANL-CESAR/XSBench. Will be compiled as c-style, so if your kernel is in C, all you need to do 
is include it; if your kernel is in C++/CUDA, you will need to wrap the include in an extern "C" block, like so:

extern "C" 
{
	#include <vexs_io_functions.h>
}

*/
extern "C" {
#include <stdio.h>
#include <string.h>

/*
Prints a border
*/
void border_print(void)
{
	printf(
	"==================================================================="
	"=============\n");
}

/*
Prints Section titles in center of 80 char terminal
*/
void center_print(const char *s, int width)
{
	int length = strlen(s);
	int i;
	for (i=0; i<=(width-length)/2; i++)
	{
		fputs(" ", stdout);
	}
	fputs(s, stdout);
	fputs("\n", stdout);
}

/*
Prints comma separated integers - for ease of reading
*/
void fancy_int( long a )
{
	if( a < 1000 )
		printf("%ld\n",a);

	else if( a >= 1000 && a < 1000000 )
		printf("%ld,%03ld\n", a / 1000, a % 1000);

	else if( a >= 1000000 && a < 1000000000 )
		printf("%ld,%03ld,%03ld\n",a / 1000000,(a % 1000000) / 1000,a % 1000 );

	else if( a >= 1000000000 )
		printf("%ld,%03ld,%03ld,%03ld\n",
			   a / 1000000000,
			   (a % 1000000000) / 1000000,
			   (a % 1000000) / 1000,
			   a % 1000 );
	else
		printf("%ld\n",a);
}
}

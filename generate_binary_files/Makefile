# What compiler we want to use to compile VEXS
VEXSCOMPILER = g++

# What compiler we want to use to link everything together
FINALCOMPILER = g++

# Flags to pass to the compiler for VEXS
VEXSFLAGS = -std=c++11 -O3 

#Flags to pass as we are linking everything together
FINALFLAGS = -fopenmp -std=gnu99 -flto -O3

#Figures out where we are located so that we can compile everything (mostly) without hardcoded file paths
VEXSDIR = $(shell pwd | sed 's/\(.*VEXS[^\/]*\).*/\1/')

#Parser source
parser = $(VEXSDIR)/parser

#Directory where we store object files
OBJ_DIR := objects

$(shell mkdir -p $(OBJ_DIR))

#parser objects
parserobj := $(patsubst $(parser)/%.cpp,$(OBJ_DIR)/%.o, $(wildcard $(parser)/*.cpp))
#parserobj += $(patsubst $(parser)/%.c,$(OBJ_DIR)/%.o, $(wildcard $(parser)/*.c))

#Directory where we collect the object files
OBJ_FILES := $(parserobj) objects/call_kernels.o

#Link together everything into one executable
VEXS: $(OBJ_FILES)
	$(FINALCOMPILER) $(FINALFLAGS) -o $@ $^

#Build parser
$(OBJ_DIR)/%.o: $(parser)/%.cpp
	$(VEXSCOMPILER) $(VEXSFLAGS) -c $< -o $@

objects/call_kernels.o : call_kernels.cpp
	$(VEXSCOMPILER) -I$(VEXSDIR)/parser $(VEXSFLAGS) -c $< -o $@

clean:
	rm VEXS objects/*.o





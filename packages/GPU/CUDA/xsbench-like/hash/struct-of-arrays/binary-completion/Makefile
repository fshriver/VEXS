# What compiler we want to use to compile VEXS
VEXSCOMPILER = nvcc

# What compuler we want to use to compile the kernels we're testing
KERNELCOMPILER = nvcc

# What compiler we want to use to link everything together
FINALCOMPILER = nvcc

# Extensions of the kernel we want to compile (otherwise compiler may try to compile header files)
kernelextensions = .cu

# GPU architecture to target. Recommended values:
# K20/K40 -- sm_35
# P100    -- sm_60
# V100    -- sm_70
SM ?= sm_60

# Other compiler options
CACHE ?= cg
ENERGY ?= detailed
OPT ?= 3

# Flags to pass to the compiler for VEXS
VEXSFLAGS = --std=c++11 -dc -Wno-deprecated-gpu-targets -Xcompiler -O3

# Flags to pass to the compiler for the kernels
KERNELFLAGS = -arch=$(SM) --std=c++11 -dc -Wno-deprecated-gpu-targets -Xptxas=-dlcm=${CACHE} -D${ENERGY} -Xptxas=-O${OPT}

#Flags to pass as we are linking everything together
FINALFLAGS = -arch=$(SM) --std=c++11 -Xptxas=-dlcm=${CACHE} -Xptxas=-O${OPT}

#Figures out where we are located so that we can compile everything (mostly) without hardcoded file paths
VEXSDIR = $(shell pwd | sed 's/\(.*VEXS[^\/]*\).*/\1/')

#Kernel source
kernels = kernels

#Parser source
parser = $(VEXSDIR)/parser

#Directory where we store object files
OBJ_DIR := objects

$(shell mkdir -p $(OBJ_DIR))

#parser objects
parserobj := $(patsubst $(parser)/%.cpp,$(OBJ_DIR)/%.o, $(wildcard $(parser)/*.cpp))

#Kernel objects
kernelobj := $(patsubst $(kernels)/%$(kernelextensions),$(OBJ_DIR)/%.o, $(wildcard $(kernels)/*$(kernelextensions)))

#Directory where we collect the object files
OBJ_FILES := $(parserobj) $(kernelobj) objects/call_kernels.o

VEXS: $(OBJ_FILES)
	$(FINALCOMPILER) $(FINALFLAGS) -o $@ $^

#Build parser
$(OBJ_DIR)/%.o: $(parser)/%.cpp
	$(VEXSCOMPILER) $(VEXSFLAGS) -c $< -o $@

objects/call_kernels.o : call_kernels.cpp
	$(VEXSCOMPILER) -I$(VEXSDIR)/parser $(VEXSFLAGS) -c $< -o $@

#Build kernels (may need to add in optional arguments later)
$(OBJ_DIR)/%.o: $(kernels)/%$(kernelextensions)
	$(KERNELCOMPILER) -I$(VEXSDIR)/parser $(KERNELFLAGS) $< -o $@

clean:
	rm VEXS objects/*.o





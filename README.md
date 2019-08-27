```
================================================================================
             _        _    _____      __       __   _______
            \  \    /  /  |  ___|    \  \    /  /  /  ____|
             \  \  /  /   | |___      \  \  /  /    \ `__.
              \  \/  /    |     |      \  \/  /      `__.  \
               \    /     |  ___|      /  /\  \          \  |
                \  /      | |___      /  /  \  \    /\__/   /
                 \/       |_____|    /__/     \__\  \______/

================================================================================
```

A collaboration between the Future Technologies Group at Oak Ridge National Laboratory (https://csmd.ornl.gov/group/future-technologies) and the FAMMoS Group at the University of Florida (http://watson.mse.ufl.edu).

Primary developers:  
Forrest Shriver (joel397@ufl.edu)  
Seyong Lee (lees2@ornl.gov)  
Steven Hamilton (hamiltonsp@ornl.gov)

# What is it?  
VEXS is a variant of the XSBench mini-app (https://github.com/ANL-CESAR/XSBench) that specifically tries to capture finer details of the memory and compute model that XSBench, by necessity, approximates. It is targeted specifically at performance analysis and prediction on platforms where more limited memory and compute resources might make the hardware more sensitive to approximations and assumptions that CPU-based platforms may not be sensitive to.

# Quickstart
To get started: 
```
cd VEXS/
./unpack_all_libraries.script
cd generate_binary_files/
./runme.script
```
This will unpack all data libraries to their appropriate places under the VEXS/ directory, and will run a reduced version of VEXS to create binary versions of all data files.

To compile and run:

```
cd VEXS/packages/GPU/CUDA/xsbench-like/naive/struct-of-arrays/
make -j16
./VEXS -i detailed_model_depleted_core -b --gputhreads 256 --gpublocks 4096
```

This launches the Struct-of-Arrays naive (basic binary) search kernel by compiling and linking together both the local kernel files located under kernels/ and the parser backend located under VEXS/parser/.

All kernels in VEXS are designed, compiled and run like above: Implementations of a specific cross-section lookup algorithm are located in their own separate folders, allowing the user to focus on specific algorithms in isolation without needing to deal with functions or code contamination from other implementatons.

# General guidelines
The VEXS codebase is essentially divided into two parts: the parser backend, located under VEXS/parser/, and the kernels that will be studied, located under VEXS/packages/. Compiling is as simple as running 'make' in the kernel folder one wishes to run. The idea is that a user will never need to touch the parser backend, instead being able to limit themselves to only the code that is located in the kernels folder they wish to examine. To avoid code duplication, the parser backend is designed to launch a call_kernels() function, which is located in the same kernel folder. This call_kernels() function subsequently calls any parser backend functions that are specifically required by the kernel under use. In this way, we can implement machinery to accomodate different lookup schemes without having this different machinery interfere with each other, and without requiring different command-line options be implemented to accomodate new features.

# Kernel guide

```packages/GPU/CUDA/xsbench-like/naive/struct-of-arrays: ``` Struct-of-Arrays version of the basic binary search kernel.  
```packages/GPU/CUDA/xsbench-like/hash/struct-of-arrays/binary-completion: ``` Struct-of-Arrays version of the hash-accelerated search, with a binary search used to complete the reduced search window.  
```packages/GPU/CUDA/xsbench-like/hash/struct-of-arrays/linear-completion: ``` Struct-of-Arrays version of the hash-accelerated search, with a linear search used to complete the reduced search window.  

# Program Options
```
--help,-h : Outputs this help message.
--input,-i : Directory under the DATA folder where VEXS will look for library data. Default path: DATA/test
--write-binary,-w : Write library data to binary file for faster reading in the future.
--binary-b : Load library data from a binary file as opposed to reading from a text file.
--lookups,-l : Total number of macroscopic cross-section lookups to perform.
--hashbins,-h : Number of hash-bins to use in the hash-accelerated search.
--openmpthreads : Number of OpenMP threads to use. Only applicable for OpenMP-specific kernels.
--gpublocks : Number of blocks to use for GPU-specific kernels.
--gputhreads: Number of threads per block to use for GPU-specific kernels.
```

# Why not just merge with XSBench?
A significant amount of work has gone into creating data libraries that represent synthetic data in a manner that is closer to real-life scenarios, which requires significantly more storage (~91 MB compressed, ~500 MB uncompressed) compared to XSBench's minimal usage of memory. Additional parsing infrastructure is also required to accomodate this significantly different dataset, which has also been engineered to accomodate VEXS' unique method of kernel construction (more on this in the wiki). As a result, establishing these various changes as a merge to the XSBench respository is infeasible, as the two applications are significantly diverged in methods and current codebases.

# parallelDMRG
Real-space parallel density matrix renormalization group (DMRG) based on ITensor

# Installing

**Currently this code requires the version 2.x (v2 git branch) series of the C++ implementation of ITensor.** If you are interested in upgrading this code to become compatible with the 3.x series, please contact support@itensor.org if you have technical questions, and consult the [version 2 to 3 upgrade guide](http://itensor.org/docs.cgi?vers=cppv3&page=upgrade2to3).

To compile the sample code (pdmrg.cc), create your own local copy of the sample Makefile (Makefile.sample) provided. Edit the ITENSOR_LIBRARY_DIR variable to point to where ITensor is located on your computer. Edit MPI_CFLAGS and MPI_LFLAGS to reflect where the MPI libraries are installed on your computer as well. For the MPI flags, you can often use a command such as "mpicxx --showme" to get a printout of the correct compiler flags for your system. Once your Makefile variables are properly set, just run "make" to compile the sample code. 

The main purpose of this software is to provide the header "parallel_dmrg.h" which you can include into your own driver code (modeled on pdmrg.cc) to call the parallel_dmrg routines for your own applications.

# Debugging Parallel Codes

ITensor includes a helpful function called `parallelDebugWait` that you call
by passing your MPI environment object. For optimized builds it just prints
the process IDs of each of your MPI processes. But for debug builds, it pauses
after printing the process IDs. At this point you can use the "attach" feature
of your debugger (such as gdb or lldb) to attach to these running processes 
in separate terminal windows. Finally, make an empty file named `GO` in the
same directory that your code is running in. The program will see this file
and begin running. After the program starts running and a bug is encountered, 
you can see the backtrace in the debugger attached to whichever process encountered the bug.

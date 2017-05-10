# parallelDMRG
Real-space parallel density matrix renormalization group (DMRG) based on ITensor

Currently must be compiled with the develop branch of ITensor.

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

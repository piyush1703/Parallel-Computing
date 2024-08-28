# Parallel-Computing

We implement and optimise a halo-exchange algorithm using Message Passing Interface (MPI) demonstrating the ability to optimally exchange boundary values of a matrix divided into multiple processors. This 2D case can be easily extended to the 3D case.

### Motivation

The primary motivation is the optimisation of scientific computations such as local gradient calculations and interpolation. Parallelisation using domain decomposition of such problems poses a significant issue in calculating values at the boundary of each domain present on separate processes. This requires some method of communicating required values to neighbouring processes. Since this is a potential bottleneck upon scaling for larger computing systems, optimisation is crucial for efficient usage of resources.

### Implementation

We implement an algorithm termed "Halo-exchange" which communicates boundary values to neighbouring processes. The problem of averaging over a 2D stencil at each point in a matrix is addressed here. Again, this exchange can be easily utilised for other issues in scientific computing. The precise problem statement can be found in the `Part 1 - Halo Exchange` directory.

### Optimisation

This exchange is often a bottleneck in scaling, so we implement an important optimisation. We utilise the fact that inter-node communications have a considerably larger latency than intra-node communications. And a naive _Halo-exchange_ uses multiple inter-node communications. We can reduce this latency using a **hierarchical implementation** where the number of inter-node exchanges is minimized. Details can be found in the `Part 2 - Optimization` directory.

The hierarchical implementation is accompanied by asynchronous `MPI_Isend` and `MPI_Irecv`. A combination of `MPI_Pack` and `MPI_Unpack` along with custom MPI vector datatypes is also used.

### Scaling Study

We used IIT Kanpur's HPC2010 cluster to scale and compare the naive and optimised implementations of the algorithm to analyse performance.

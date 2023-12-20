# Usage Instructions
What installation steps are necessary (if any)?
How do we run the program and see output? (please include a few demo scripts!)

To build, run `make`.

Usage:

`./bin/rubik -d [depth] -n [num_cubes] [-i] [-m] [-g]`

Generates `num_cubes` cubes, scrambled to a depth of `depth`, and solves each
with each of the flagged solvers, timing the solve. 

`-i` Enables the IDDFS solver,
which uses iterative deepening depth-first-search on CPU. The slowest of the
solvers, only recommended to a depth of 6. 

`-m` Enables the Manhattan solver,
which uses IDA* with a manhattan distance heuristic on CPU. The fastest CPU
solver, recommended to a depth of 12. 

`-g` Enables the GPU-Manhattan solver, which
uses GPU-IDA* (see project description) with the manhattan distance metric.
Recommended to a depth of 15.

# Project Description

Existing algorithmic Rubik's cube solvers rely on huge (> 1 GB) pattern tables
of pre-computed distances to a solved cube (or subgroup) to guide their search.
So-called "naive" methods which do not rely on pattern tables, or rely on small
(< 1 MB) pattern tables to guide their search have only shown success in solving
cubes to a move depth of 11. This is primarily a result of the fact that most
people are not interested in this problem, but nonethless, the goal of this
project was to leverage the GPU to accelerate Rubik's cube search algorithms
without large and unwieldy pattern databases.

To that end, we implemented a version of iterative-deepening A* (IDA*), the
standard search algorithm for Rubik's cube solvers, with block-level
parallelism, following a paper (https://arxiv.org/pdf/1705.02843.pdf) which
presents a version of this algorithm for solving sliding-block puzzles on GPU.
We additionally implement a small pattern table in order to use the manhattan
distance metric on GPU.

## GPU-IDA*
Standard IDA* proceeds like iterative deepening, but with a heuristic function
h(cube) which acts as a lower bound on the number of moves remaining to solve
the cube. A global bound is initialized to h(initial cube), and a depth-first
search is performed, cutting off any branch whose cost = depth + h(node) is
greater than the global bound. When all nodes have been explored and no solution
found, the global bound is updated to the minimum cost of all leaf nodes (i.e.
nodes with cost > the previous bound).

The naive method of parallelizing IDA* is to run the algorithm on CPU until you
reach an iteration where the number of leaf nodes explored is greater than some
preset bound, and then delegate each of those leaf nodes (i.e. subproblems) to a
GPU thread. This technique, however, has many disadvantages:
- Some threads may finish their subproblems much faster than others, leading to
  thread stalling.
- Incredibly high warp divergence leads to slow, nearly serial processing times
As a result, thread-based parallelization almost never shows improvement over
a concurrent version of the same solver.

We instead rely on a block-parallel approach. Each subproblem is assigned to a
block, which contains 24 threads. A block contains a shared search frontier,
which we implement as a standalone stack module (struct stack in search_gpu.cu).
The stack implements the following operations:
- stack_pop: Pops 2 elements from the stack in parallel, giving the first to
  threads 0 - 11 and the second to threads 12 - 23
- stack_push: Pushes an element (happens concurrently)

During the IDA* search, each thread calls stack_pop to receive a cube to
process, and then picks a move to execute on that cube from its thread index mod
12 (the number of Rubik's cube moves). If the new cube's cost is less than the
global bound, it gets placed onto the stack. Otherwise, we update a global
"probable bound" with an atomicMin which is returned from the kernel to become
the bound in the next iteration if no solution is found. In this way, the work
exploring of a subproblem is effectively split over the block - warp divergence
is minimized, and all threads have work until a solution is found.

## Manhattan Metric Pattern Table
The Manhattan metric was the heuristic function used in the Manhattan solver and
GPU-Manhattan solver. This metric is based on the number of moves required to
move each corner and edge cubie individually to its solved position (this is
done naively, ignoring how it affects other cubies, for each cubie). This cost
is ensured to be admissible in order to preserve the optimality of the solution.

During IDA*, the heuristic is used to estimate a probable lower bound on the
number of moves required to solve a given subproblem. As this is done during the
evaluation of every move, it must be a low-cost operation. To accomodate this,
we store the number of moves required to "solve" any given cubie in any given
position in a "pattern table". This pattern table is precomputed by the CPU before
the GPU subproblems are generated. To do so, the CPU performs an iterative
deepening depth-first search for every possible cubie's position and
orientation. This is done by having the solver find the solution depth required
to move the given cubie to that position and orientation (the same moves can be
executed in reverse to return the cubie to its solved configuration). This is
possible because IDDFS is IDA* with h(cube) = 0, meaning the searcher will
uniformly deepen the tree until a solution is found (in practice, to a depth of
3). The Manhattan metric pattern tables are then moved into GPU global memory
and are accessed as required during IDA*.

# Results / Performance Analysis

For lower depths (<8), the solution is often found on the CPU before enough
subproblems have been gathered to dispatch to the GPU, and otherwise is found
very quickly on GPU. In these cases, the overhead mostly comes from the cost of
spinning up a kernel, and so the CPU shows better performance. But at higher
depths, the GPU performs much better than the CPU, despite exploring
approximately the same number of nodes. In the table below we show the average
time for each algorithm to find a solution at a given depth. Note that this is
distinct from the scramble depth - often, a cube scrambled in n moves can be
solved in ~n-2 because some moves cancel each other out.

```
+-------------------+-----------+---------------+---------------+
| Depth of Solution |           Avg. Wall Time (s)              |
+-------------------+-----------+---------------+---------------+
|                   | IDDFS-CPU | Manhattan-CPU | Manhattan-GPU |
+-------------------+-----------+---------------+---------------+
|                 6 |       7.6 |       0.00030 |       0.00302 |
|                 7 |      61.4 |       0.00323 |       0.00468 |
|                 8 |       N/A |        0.0335 |       0.00829 |
|                 9 |       N/A |         0.071 |       0.00821 |
|                10 |       N/A |          1.36 |        0.0253 |
|                11 |       N/A |          9.61 |         0.114 |
|                12 |       N/A |         89.10 |          1.28 |
|                13 |       N/A |        604.18 |          13.4 |
+-------------------+-----------+---------------+---------------+
```

At depths of 10 - 13, the GPU has a speedup of roughly 50x-100x over the CPU
version. At a depth of 13, the manhattan algorithms are exploring almost exactly
a billion nodes on average. meaning that the CPU version is able to explore 1.6
million nodes per second while the GPU version can explore 75 million nodes per
second.

## Limitations
One limitation of our GPU algorithm is that the stack size grows arbitrarily
with depth. This is helped quite a bit by pruning with the Manhattan metric, but
we are still somewhat close to shared memory limits - based on experimentation
we chose a maximum stack size of 400, and our cube state struct is 50 bytes,
meaning we're using 20kB, or slightly less than half of shared memory. With more
compute at higher depths, this might become a limiting factor.

Additionally, our GPU search is limited as it requires the number of threads in
a block to be a multiple of the number of moves (12), meaning we have 8 threads
per warp which are sitting idle. Future work could find a better way to
distribute the work between threads such that an arbitrary number of threads
could be used, but it is not clear how this would be possible given our
one-move-per-thread paradigm - this would mainly be a question of a more complex
stack_pop implementation to handle the load balancing.

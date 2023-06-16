#include <algorithm>
#include <iostream>

#include "heuristic.hpp"
#include "search_gpu.cuh"

/*
  Note:
  Many GPU IDA* data structures are GPU versions of CPU IDA* structures, as we needed the
  structures available in both. For documentation on these structures, see the corresponding
  documentation in search.hpp/cpp, rubiks.hpp/cpp, and heuristic.hpp/cpp, as specified.
*/

#define gpuErrchk(ans) \
  { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true) {
  if (code != cudaSuccess) {
    fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
    if (abort) exit(code);
  }
}

#define NUM_BLOCKS 200 // Minimum number of leaf nodes required ahead of block-parallelized IDA*

const uint8_t FOUND = 0;
const uint8_t NOT_FOUND = UINT8_MAX;

__constant__ uint8_t cornerPosGPU[8][3] = {{6, 10, 16},  {4, 18, 24},  {2, 26, 32},  {0, 8, 34},
                                           {42, 30, 20}, {40, 22, 12}, {46, 14, 36}, {44, 28, 38}};

__constant__ uint8_t edgePosGPU[12][3] = {{5, 17},  {7, 9},   {3, 25},  {1, 33},
                                          {41, 21}, {47, 13}, {43, 29}, {45, 37},
                                          {23, 11}, {19, 31}, {35, 15}, {39, 27}};

__constant__ uint8_t colorFlagsGPU[6] = {0b10000000, 0b01000000, 0b00100000,
                                         0b00010000, 0b00001000, 0b00000100};

__constant__ uint8_t FOUND_GPU = 0;
__constant__ uint8_t NOT_FOUND_GPU = UINT8_MAX;

/*
  Subproblems are branches of the search tree which were pruned at a specified
  bound but should be subsequently explored by blocks.
  The host version of this structure includes vector types for ease of insertion,
  while the device version converts takes the last cube of this vector as a static
  array to decrease the runtime.
*/
struct subproblem_device {
  uint8_t cube[48];
  uint8_t lastMove;
  uint8_t curDepth;
};
struct subproblem_host {
  vector<RubiksCube> cubePath;
  vector<RubiksCube::Rotation> movePath;
  uint8_t curDepth;
  uint8_t bound;
};

/*
  Rubiks cube implementation - device version.
  See rubiks.hpp/cpp for details.
*/
__device__ uint8_t heuristic(uint8_t *cube) {
  return 0;
}

__device__ uint64_t getFace(uint8_t *cube, uint8_t f) {
  return *(uint64_t *)&cube[(unsigned)f * 8];
}

__device__ bool is_solved(uint8_t *cube) {
  return getFace(cube, 0) == 0x00000000 && getFace(cube, 1) == 0x101010101010101 &&
         getFace(cube, 2) == 0x202020202020202 && getFace(cube, 3) == 0x303030303030303 &&
         getFace(cube, 4) == 0x404040404040404 && getFace(cube, 5) == 0x505050505050505;
}

__device__ uint8_t getEdge(uint8_t *cube, uint8_t pos) {
  uint8_t c1 = cube[edgePosGPU[pos][0]];
  uint8_t c2 = cube[edgePosGPU[pos][1]];

  uint8_t cubie = (colorFlagsGPU[(uint8_t)c1] | colorFlagsGPU[(uint8_t)c2]);
  uint8_t orientation;
  switch (cubie) {
    case 0b10100000:
    case 0b11000000:
    case 0b10010000:
    case 0b10001000:
      orientation = !(c1 == 0);
      break;
    case 0b00100100:
    case 0b01000100:
    case 0b00010100:
    case 0b00001100:
      orientation = !(c1 == 5);
      break;
    case 0b01100000:
    case 0b00110000:
      orientation = !(c1 == 2);
      break;
    case 0b01001000:
    case 0b00011000:
      orientation = !(c1 == 4);
      break;
    default:
      break;
  }
  return (uint8_t)cubie | orientation;
}

__device__ uint8_t getCorner(uint8_t *cube, uint8_t pos) {
  uint8_t c1 = cube[cornerPosGPU[pos][0]];
  uint8_t c2 = cube[cornerPosGPU[pos][1]];
  uint8_t c3 = cube[cornerPosGPU[pos][2]];

  uint8_t cubie =
      (colorFlagsGPU[(uint8_t)c1] | colorFlagsGPU[(uint8_t)c2] | colorFlagsGPU[(uint8_t)c3]);
  uint8_t orientation;

  switch (cubie) {
    case 0b11100000:
      orientation = (c1 == 2) ? 2 : (c1 == 1);
      break;
    case 0b10110000:
      orientation = (c1 == 3) ? 2 : (c1 == 2);
      break;
    case 0b10011000:
      orientation = (c1 == 4) ? 2 : (c1 == 3);
      break;
    case 0b11001000:
      orientation = (c1 == 4) ? 2 : (c1 == 1);
      break;
    case 0b00110100:
      orientation = (c1 == 2) ? 2 : (c1 == 3);
      break;
    case 0b01100100:
      orientation = (c1 == 1) ? 2 : (c1 == 2);
      break;
    case 0b01001100:
      orientation = (c1 == 4) ? 2 : (c1 == 1);
      break;
    case 0b00011100:
      orientation = (c1 == 4) ? 2 : (c1 == 3);
      break;
    default:
      break;
  }
  return orientation | (uint8_t)cubie;
}

__device__ bool is_move_opposite(uint8_t m1, uint8_t m2) {
  return m1 >> 1 == m2 >> 1 && m1 != m2;
}

__device__ inline void turnClockwise(uint8_t *cube, uint8_t f) {
  uint64_t face = *(uint64_t *)&cube[(unsigned)f * 8];
  face = (face << 16) | (face >> 48);
  *(uint64_t *)&cube[(unsigned)f * 8] = face;
}

__device__ inline void turnCounterClockwise(uint8_t *cube, uint8_t f) {
  uint64_t face = *(uint64_t *)&cube[(unsigned)f * 8];
  face = (face >> 16) | (face << 48);
  *(uint64_t *)&cube[(unsigned)f * 8] = face;
}

__device__ inline void updateSides(uint8_t *cube, unsigned s0, unsigned s1, unsigned s2,
                                   unsigned s3, unsigned c0, unsigned c1, unsigned c2,
                                   unsigned c3) {
  uint16_t s3_orig = *((uint16_t *)&cube[s3]);

  *((uint16_t *)&cube[s3]) = *((uint16_t *)&cube[s2]);
  *((uint16_t *)&cube[s2]) = *((uint16_t *)&cube[s1]);
  *((uint16_t *)&cube[s1]) = *((uint16_t *)&cube[s0]);
  *((uint16_t *)&cube[s0]) = s3_orig;

  uint8_t c3_orig = cube[c3];
  cube[c3] = cube[c2];
  cube[c2] = cube[c1];
  cube[c1] = cube[c0];
  cube[c0] = c3_orig;
}

__device__ void cube_rotate(uint8_t *cube, uint8_t rotation) {
  switch (rotation) {
    case 0:
      turnClockwise(cube, 0);
      updateSides(cube, 32, 24, 16, 8, 34, 26, 18, 10);
      break;
    case 1:
      turnCounterClockwise(cube, 0);
      updateSides(cube, 8, 16, 24, 32, 10, 18, 26, 34);
      break;
    case 2:
      turnClockwise(cube, 5);
      updateSides(cube, 12, 20, 28, 36, 14, 22, 30, 38);
      break;
    case 3:
      turnCounterClockwise(cube, 5);
      updateSides(cube, 36, 28, 20, 12, 38, 30, 22, 14);
      break;
    case 4:
      turnClockwise(cube, 1);
      updateSides(cube, 6, 22, 46, 34, 0, 16, 40, 36);
      break;
    case 5:
      turnCounterClockwise(cube, 1);
      updateSides(cube, 34, 46, 22, 6, 36, 40, 16, 0);
      break;
    case 6:
      turnClockwise(cube, 3);
      updateSides(cube, 38, 42, 18, 2, 32, 44, 20, 4);
      break;
    case 7:
      turnCounterClockwise(cube, 3);
      updateSides(cube, 2, 18, 42, 38, 4, 20, 44, 32);
      break;
    case 8:
      turnClockwise(cube, 2);
      updateSides(cube, 30, 40, 10, 4, 24, 42, 12, 6);
      break;
    case 9:
      turnCounterClockwise(cube, 2);
      updateSides(cube, 4, 10, 40, 30, 6, 12, 42, 24);
      break;
    case 10:
      turnClockwise(cube, 4);
      updateSides(cube, 26, 0, 14, 44, 28, 2, 8, 46);
      break;
    case 11:
      turnCounterClockwise(cube, 4);
      updateSides(cube, 44, 14, 0, 26, 46, 8, 2, 28);
      break;
    default:
      break;
  }
}

/*
  Manhattan heuristic implementation - device version.
  See heuristic.hpp/cpp for details.
*/
__device__ uint8_t manhattan_heuristic_gpu(uint8_t *cube, uint8_t *manhattan_corners,
                                           uint8_t *manhattan_edges) {
  uint8_t corner_total = 0;
  uint8_t edge_total = 0;

  for (size_t i = 0; i < 8; i++) {
    corner_total += manhattan_corners[255 * i + getCorner(cube, i)];
  }
  for (size_t i = 0; i < 12; i++) {
    edge_total += manhattan_edges[255 * i + getEdge(cube, i)];
  }
  // std::cout << (int)max(corner_total / 4, edge_total / 4) << "\n";
  return max(corner_total / 4, edge_total / 4);
}

/*
  Thread-safe stack implementation.
*/

#define NUM_TURNS 12 // Number of possible moves on a cube.

/*
  BLOCK_DIM was set to 24 due to being the largest multiple of NUM_TURNS
  less than the number of threads in a warp. As a result, all possible moves can
  be evaluated on 2 cubes at a time without requiring explicit synchronization.
*/
#define BLOCK_DIM 24

#define MAX_STACK_LEN 450

/* Stack structure, including a size and buffer. */
struct stack {
  unsigned int n;
  subproblem_device buf[MAX_STACK_LEN];
};

/* Returns whether the stack is empty. */
__device__ static inline bool stack_is_empty(stack *stack) {
  bool ret = (stack->n == 0);
  __syncthreads();
  return ret;
}

/* Thread-safe stack put. */
__device__ static inline void stack_put(stack *stack, subproblem_device *state, bool put) {
  if (put) {
    unsigned int i = atomicInc(&stack->n, UINT_MAX);
    stack->buf[i] = *state;
  }
  __syncthreads();
}

/*
  Pops BLOCK_DIM/NUM_TURNS subproblem_device elements from the top of the stack,
  storing each in each thread's provided subproblem_device *. Each thread evaluates
  one of the 12 moves on this subproblem.
*/
__device__ static inline bool stack_pop(stack *stack, subproblem_device *state) {
  int tid = threadIdx.x; // [0, BLOCK_DIM - 1]
  int i = (int)stack->n - 1 - (int)(tid / NUM_TURNS);
  if (i >= 0) *state = stack->buf[i];
  __syncthreads();
  if (tid == 0) stack->n = stack->n >= BLOCK_DIM / NUM_TURNS ? stack->n - BLOCK_DIM / NUM_TURNS : 0;
  __syncthreads();
  return i >= 0;
}

/*
  GPU Search Functions
*/

/*
  Performs DFS search given a certain bound, adding all pruned branches to `host_list` as
  subproblems. Returns FOUND if a path is found or the cost of the minumum-cost pruned branch if not
  found.
*/
uint8_t subproblem_helper(vector<RubiksCube> &cubePath, vector<RubiksCube::Rotation> &movePath,
                          uint8_t curdepth, uint8_t bound, vector<subproblem_host> &host_list) {
  RubiksCube cube_curr = cubePath.back();
  uint8_t cost = manhattan_heuristic(cube_curr) + curdepth;
  if (cost > bound) {
    // Leaf node - add to subproblems list
    struct subproblem_host subprob;
    subprob.cubePath = cubePath;
    subprob.movePath = movePath;
    subprob.bound = bound;
    subprob.curDepth = curdepth;
    // std::cout << "Adding to list: " << (int)cost << " " << (int)bound << "\n";
    host_list.push_back(subprob);
    return cost;
  }
  if (cube_curr.isSolved()) {
    return FOUND;
  }
  uint8_t min = NOT_FOUND;
  for (uint8_t rot = 0; rot < (uint8_t)RubiksCube::Rotation::LAST_MOVE; rot++) {
    RubiksCube::Rotation move = (RubiksCube::Rotation)rot;
    RubiksCube succ = RubiksCube(cube_curr);
    succ.rotate(move);
    if (std::find(cubePath.begin(), cubePath.end(), succ) == cubePath.end()) {
      cubePath.push_back(succ);
      movePath.push_back(move);
      uint8_t probable_bound =
          subproblem_helper(cubePath, movePath, curdepth + 1, bound, host_list);
      if (probable_bound == FOUND) {
        return FOUND;
      }
      if (probable_bound < min) {
        min = probable_bound;
      }
      cubePath.pop_back();
      movePath.pop_back();
    }
  }
  return min;
}

/*
  Performs IDA* on the provided cube until reaching a specified threshhold of pruned branches
  (NUM_BLOCKS). The leaves of these pruned branches are stored as subproblems to be searched by the
  blocks. This is performed on the CPU. If a solution is found over the course of subproblem
  generation, the function returns immediately, with the depth of the found solution stored in
  `bound`.
*/
__host__ vector<subproblem_host> generateSubproblems(RubiksCube &cube, uint8_t *bound) {
  vector<RubiksCube> cubePath;
  vector<RubiksCube::Rotation> movePath;
  vector<subproblem_host> subproblems_host_list;
  *bound = manhattan_heuristic(cube);
  cubePath.push_back(cube);
  while (true) {
    subproblems_host_list.clear();
    uint8_t probable_bound =
        subproblem_helper(cubePath, movePath, 0, *bound, subproblems_host_list);
    if (probable_bound == FOUND) {
      subproblems_host_list.clear();
      return subproblems_host_list;
    }
    if (subproblems_host_list.size() >= NUM_BLOCKS) {
      return subproblems_host_list;
    }
    *bound = probable_bound;
  }
}

/*
  Each thread handles one move on an individual subproblem, popped from the stack.
  If the actual cost remains less than the specified `bound`, the resulting subproblem
  is pushed onto the top of the stack to be evaluated by the block later.
  The bound used by all blocks (`global_probable_bound`) is kept updated to the lowest
  bound across all pruned branches.

  Moves which immediately reverse the last move are not evaluated.
*/
__device__ static void gpu_search_kernel_helper(struct stack *pStack, uint8_t global_bound,
                                                int *global_probable_bound,
                                                struct subproblem_device *solution,
                                                uint8_t *h_patterntable_corners,
                                                uint8_t *h_patterntable_edges) {
  subproblem_device cur;
  uint8_t bound = global_bound;
  uint8_t probable_bound = NOT_FOUND_GPU;
  __syncthreads();
  for (;;) {
    if (stack_is_empty(pStack) || *global_probable_bound == FOUND_GPU) {
      break;
    }
    bool put = false;
    bool success = stack_pop(pStack, &cur); // Takes out 2 elements

    if (success) {
      uint8_t move = threadIdx.x % 12;
      if (is_move_opposite(cur.lastMove, move)) { // Do not immediately undo a move.
        continue;
      }
      cube_rotate(cur.cube, move);

      cur.curDepth += 1;
      cur.lastMove = move;
      uint8_t cost =
          manhattan_heuristic_gpu(cur.cube, h_patterntable_corners, h_patterntable_edges) +
          cur.curDepth;
      if (cost <= bound) {
        if (is_solved(cur.cube)) {
          *global_probable_bound = FOUND_GPU;
          *solution = cur;
          break;
        }

        put = true;
      } else {
        if (cost < probable_bound) {
          probable_bound = cost;
        }
      }
    }
    stack_put(pStack, &cur, put); // Add if put is true.
  }
  if (threadIdx.x == 0) {
    atomicMin(global_probable_bound, (int)probable_bound); // Update bound used by all blocks.
  }
}

/*
  Serves as the root call to gpu_search_kernel_helper, copying the root subproblem
  for the block and pushing this copy onto the stack in one thread before calling.
*/
__global__ void gpu_search_kernel(struct subproblem_device *subproblem_list, size_t n_subproblems,
                                  uint8_t global_bound, int *global_probable_bound,
                                  struct subproblem_device *solution,
                                  uint8_t *h_patterntable_corners, uint8_t *h_patterntable_edges) {
  __shared__ struct stack pStack;
  int tid = threadIdx.x;
  int bid = blockIdx.x;
  subproblem_device root = subproblem_list[bid];

  subproblem_device cur;
  for (int i = 0; i < 48; i++) {
    cur.cube[i] = root.cube[i];
  }
  cur.curDepth = root.curDepth;
  cur.lastMove = root.lastMove;

  if (tid == 0) {
    pStack.buf[0] = cur;
    pStack.n = 1;
  }

  __syncthreads();
  gpu_search_kernel_helper(&pStack, global_bound, global_probable_bound, solution,
                           h_patterntable_corners, h_patterntable_edges);
}

/*
  Solves a Rubik's cube using block-parallelized IDA*, returning the minimal number of moves
  required to solve it.
  First uses a CPU-driven IDA* to compile a list of subproblems (pruned
  branches), handing each branch off to a block once enough subproblems (at least NUM_BLOCKS) are
  added. If the solution is found during this CPU search, the function immediately returns the depth
  of the solution.
*/
__host__ uint8_t gpu_search(RubiksCube &cube) {
  uint8_t init_bound;

  // Subproblem generation
  vector<subproblem_host> subproblems = generateSubproblems(cube, &init_bound);
  size_t n_subproblems = subproblems.size();
  if (n_subproblems == 0) {
    return init_bound;
  }

  // Conversion of subproblem_host's to subproblem_device's and cuda allocation.
  struct subproblem_device *subproblem_list;
  gpuErrchk(cudaMalloc(&subproblem_list, n_subproblems * sizeof(struct subproblem_device)));
  struct subproblem_device *cursor = subproblem_list;

  for (size_t i = 0; i < n_subproblems; i++) {
    struct subproblem_host hsub = subproblems[i];
    subproblem_device dev;
    memcpy(dev.cube, hsub.cubePath.back().get_cube_array(), 48 * sizeof(uint8_t));

    dev.curDepth = hsub.curDepth;
    dev.lastMove = (uint8_t)hsub.movePath.back();

    gpuErrchk(cudaMemcpy(cursor, &dev, sizeof(struct subproblem_device), cudaMemcpyHostToDevice));
    cursor++;
  }

  struct subproblem_device *solution;
  gpuErrchk(cudaMalloc(&solution, sizeof(struct subproblem_device)));
  int *global_probable_bound;
  gpuErrchk(cudaMalloc(&global_probable_bound, sizeof(int)));

  // Generation of manhattan distance pattern tables.
  uint8_t *h_patterntable_corners, *h_patterntable_edges;
  cudaMalloc(&h_patterntable_corners, 8 * 255 * sizeof(uint8_t));
  cudaMalloc(&h_patterntable_edges, 12 * 255 * sizeof(uint8_t));

  cudaMemcpy(h_patterntable_corners, manhattan_corners, 8 * 255 * sizeof(uint8_t),
             cudaMemcpyHostToDevice);
  cudaMemcpy(h_patterntable_edges, manhattan_edges, 12 * 255 * sizeof(uint8_t),
             cudaMemcpyHostToDevice);

  int global_bound_host = NOT_FOUND;
  uint8_t bound = init_bound;
  int global_probable_bound_val = (int)NOT_FOUND;

  // GPU IDA* search loop
  while (true) {
    gpuErrchk(cudaMemcpy(global_probable_bound, &global_probable_bound_val, sizeof(int),
                         cudaMemcpyHostToDevice));

    gpu_search_kernel<<<n_subproblems, BLOCK_DIM>>>(subproblem_list, n_subproblems, bound,
                                                    global_probable_bound, solution,
                                                    h_patterntable_corners, h_patterntable_edges);

    // Copying results of iteration to host memory, stopping if terminated
    cudaMemcpy(&global_bound_host, global_probable_bound, sizeof(int), cudaMemcpyDeviceToHost);
    if (global_bound_host == FOUND) {
      std::cout << "Found solution\n";
      break;
    }
    if (global_bound_host == NOT_FOUND) {
      std::cout << "Not found \n";
      break;
    }
    bound = (uint8_t)global_bound_host;
  }
  // Copying found depth back to host memory
  uint8_t solution_depth = 0;
  cudaMemcpy(&solution_depth, &solution->curDepth, sizeof(uint8_t), cudaMemcpyDeviceToHost);

  return solution_depth;
}

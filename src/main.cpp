#include <cuda_runtime.h>

#include <chrono>
#include <cstring>
#include <iostream>
using namespace std::chrono;
#include <stdlib.h>
#include <unistd.h>

#include "heuristic.hpp"
#include "search_gpu.cuh"
bool solve_goal(RubiksCube &cube, void *aux) {
  return cube.isSolved();
}

bool is_move_opp(uint8_t m1, uint8_t m2) {
  return m1 >> 1 == m2 >> 1 && m1 != m2;
}

void printHeader() {
  for (int i = 0; i < 25; i++) {
    std::cout << "-";
  }
}

bool parse_args(int argc, char *argv[], int *depth, int *n_cubes, bool *iddfs, bool *manhattan,
                bool *gpu) {
  int opt;
  *depth = -1;
  *n_cubes = -1;
  *iddfs = false;
  *manhattan = false;
  *gpu = false;

  while ((opt = getopt(argc, argv, ":d:n:img")) != -1) {
    switch (opt) {
      case 'd':
        *depth = atoi(optarg);
        if (*depth <= 0) {
          std::cerr << "Depth must be a positive integer" << std::endl;
          return false;
        }
        break;
      case 'n':
        *n_cubes = atoi(optarg);
        if (*n_cubes <= 0) {
          std::cerr << "n must be a positive integer" << std::endl;
          return false;
        }
        break;
      case 'i':
        *iddfs = true;
        break;
      case 'm':
        *manhattan = true;
        break;
      case 'g':
        *gpu = true;
        break;
      case ':':
        std::cerr << "Option needs a value" << std::endl;
        return false;
      case '?':
        std::cerr << "Unknown option: " << char(optopt) << std::endl;
        return false;
    }
  }
  return true;
}

int main(int argc, char *argv[]) {
  manhattan_init();
  cudaSetDevice(0);

  int scramble_depth;
  int n_cubes;
  bool iddfs;
  bool manhattan;
  bool gpu;

  parse_args(argc, argv, &scramble_depth, &n_cubes, &iddfs, &manhattan, &gpu);
  IDASearcher *man_searcher = NULL;
  IDASearcher *searcher = NULL;
  std::cout << "Generating " << n_cubes << " cubes with scramble depth " << scramble_depth
            << "...\n";
  if (manhattan) {
    man_searcher = new IDASearcher(manhattan_heuristic, solve_goal, NULL);
  }

  if (iddfs) {
    searcher = new IDASearcher(iddfs_heuristic, solve_goal, NULL);
  }

  for (int ci = 0; ci < n_cubes; ci++) {
    printHeader();
    std::cout << " Cube " << ci << " ";
    printHeader();
    std::cout << "\n";

    RubiksCube *cube = new RubiksCube;
    cube->scramble(scramble_depth);

    if (gpu) {
      auto start_gpu = high_resolution_clock::now();
      uint8_t depth = gpu_search(*cube);
      auto end_gpu = high_resolution_clock::now();
      auto duration_gpu = duration_cast<microseconds>(end_gpu - start_gpu);

      std::cout << "GPU search found solution with " << (int)depth << " moves in "
                << duration_gpu.count() << " microseconds.\n";
    }

    if (manhattan) {
      auto start_man = high_resolution_clock::now();
      vector<RubiksCube::Rotation> moves = man_searcher->search(*cube);
      auto end_man = high_resolution_clock::now();
      auto duration_man = duration_cast<microseconds>(end_man - start_man);

      std::cout << "Manhattan search found solution with " << moves.size() << " moves in "
                << duration_man.count() << " microseconds.\n";
    }

    if (iddfs) {
      auto start_man = high_resolution_clock::now();
      vector<RubiksCube::Rotation> moves2 = searcher->search(*cube);
      auto end_man = high_resolution_clock::now();
      auto duration_man = duration_cast<microseconds>(end_man - start_man);

      std::cout << "IDDFS search found solution with " << moves2.size() << " moves in "
                << duration_man.count() << " microseconds.\n";
    }
  }
}

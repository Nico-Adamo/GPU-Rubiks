#pragma once

#include <vector>

#include "rubiks.hpp"

using namespace std;

#define MAX_DEPTH 20

typedef uint8_t (*heuristic_func_t)(RubiksCube &);
typedef bool (*goal_func_t)(RubiksCube &, void *aux);

class IDASearcher {
 public:
  IDASearcher(heuristic_func_t heuristic, goal_func_t goal, void *goal_aux);
  vector<RubiksCube::Rotation> search(RubiksCube &cube);
  void setGoalAux(void *aux);

 private:
  heuristic_func_t heuristic;
  goal_func_t goal;
  void *goal_aux;
};
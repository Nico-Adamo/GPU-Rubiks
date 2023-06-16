#include "search.hpp"

#include <algorithm>
#include <iostream>

const uint8_t FOUND = 0;
const uint8_t NOT_FOUND = UINT8_MAX;

int totalConsidered = 0;

IDASearcher::IDASearcher(heuristic_func_t heuristic, goal_func_t goal, void *goal_aux) {
  this->heuristic = heuristic;
  this->goal = goal;
  this->goal_aux = goal_aux;
}

void IDASearcher::setGoalAux(void *aux) {
  this->goal_aux = aux;
}

/*
  Performs DFS search given a certain bound.
  Returns FOUND if a path is found or the cost of the minumum-cost pruned branch
  if not found.
*/
uint8_t search_helper(vector<RubiksCube> &cubePath, vector<RubiksCube::Rotation> &movePath,
                      uint8_t curdepth, uint8_t bound, heuristic_func_t heuristic, goal_func_t goal,
                      void *goal_aux) {
  totalConsidered++;
  RubiksCube cube_curr = cubePath.back();
  uint8_t cost = heuristic(cube_curr) + curdepth;
  if (cost > bound) {
    return cost;
  }
  if (goal(cube_curr, goal_aux)) {
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
      totalConsidered++;
      uint8_t probable_bound =
          search_helper(cubePath, movePath, curdepth + 1, bound, heuristic, goal, goal_aux);
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
  Peforms IDA* searches, iteratively increasing the bound on searches.
  The bound is evaluated based on the current depth and heuristic.
*/
vector<RubiksCube::Rotation> IDASearcher::search(RubiksCube &cube) {
  vector<RubiksCube> cubePath;
  vector<RubiksCube::Rotation> movePath;
  uint8_t bound = heuristic(cube);
  cubePath.push_back(cube);
  totalConsidered = 0;
  while (true) {
    uint8_t probable_bound =
        search_helper(cubePath, movePath, 0, bound, this->heuristic, this->goal, this->goal_aux);
    if (probable_bound == FOUND) {
      std::cout << "Total considered nodes: " << totalConsidered << "\n";
      return movePath;
    }
    if (probable_bound == NOT_FOUND) {
      vector<RubiksCube::Rotation> ret;
      std::cout << "NOT FOUND"
                << "\n";
      return ret;
    }
    bound = probable_bound;
  }
}

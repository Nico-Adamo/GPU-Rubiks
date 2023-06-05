#include "search.hpp"

#include <algorithm>
#include <iostream>

const uint8_t FOUND = 0;
const uint8_t NOT_FOUND = UINT8_MAX;

IDASearcher::IDASearcher(uint8_t (*heuristic)(RubiksCube &)) { this->heuristic = heuristic; }

uint8_t search_helper(vector<RubiksCube> &cubePath, vector<RubiksCube::Rotation> &movePath,
                      uint8_t curdepth, uint8_t bound, uint8_t (*heuristic)(RubiksCube &)) {
  RubiksCube cube_curr = cubePath.back();
  uint8_t cost = heuristic(cube_curr) + curdepth;
  if (cost > bound) {
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
      uint8_t probable_bound = search_helper(cubePath, movePath, curdepth + 1, bound, heuristic);
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

vector<RubiksCube::Rotation> IDASearcher::search(RubiksCube &cube) {
  vector<RubiksCube> cubePath;
  vector<RubiksCube::Rotation> movePath;
  uint8_t bound = heuristic(cube);
  cubePath.push_back(cube);
  while (true) {
    uint8_t probable_bound = search_helper(cubePath, movePath, 0, bound, this->heuristic);
    if (probable_bound == FOUND) {
      return movePath;
    }
    if (probable_bound == NOT_FOUND) {
      vector<RubiksCube::Rotation> ret;
      return ret;
    }
    bound = probable_bound;
  }
}

uint8_t iddfs_heuristic(RubiksCube &cube) { return 0; }

int main() {
  RubiksCube *cube = new RubiksCube;
  cube->scramble(30);
  cube->printCube();
  IDASearcher *searcher = new IDASearcher(iddfs_heuristic);
  vector<RubiksCube::Rotation> moves = searcher->search(*cube);
  for (auto &move : moves) {
    std::cout << (int)move << "\n";
  }
}
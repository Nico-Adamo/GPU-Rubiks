#include <bitset>
#include <cstring>
#include <iostream>

#include "search.hpp"

uint8_t manhattan_dist[8][255];
string cornerStrings[8] = {"C_WGR", "C_WRB", "C_WBO", "C_WGO", "C_YBR", "C_YRG", "C_YGO", "C_YBO"};
RubiksCube::CornerCubie corners[8] = {
    RubiksCube::CornerCubie::C_WGR, RubiksCube::CornerCubie::C_WRB, RubiksCube::CornerCubie::C_WBO,
    RubiksCube::CornerCubie::C_WGO, RubiksCube::CornerCubie::C_YBR, RubiksCube::CornerCubie::C_YRG,
    RubiksCube::CornerCubie::C_YGO, RubiksCube::CornerCubie::C_YBO};

uint8_t iddfs_heuristic(RubiksCube &cube) {
  return 0;
}

uint8_t manhattan_heuristic(RubiksCube &cube) {
  uint8_t total = 0;
  for (size_t i = 0; i < 8; i++) {
    total += manhattan_dist[i][cube.getCorner(i)];
  }
  return total / 8;
}

bool solve_goal(RubiksCube &cube, void *aux) {
  return cube.isSolved();
}

bool cornerSolver_goal(RubiksCube &cube, void *aux) {
  uint8_t pos = *((uint8_t *)aux);
  uint8_t corner = *(((uint8_t *)aux) + 1);
  return cube.getCorner(pos) == corner;
}

void fillManhattanArray() {
  memset((void *)manhattan_dist, 0, 255 * 8 * sizeof(uint8_t));
  vector<RubiksCube::Rotation> moves;
  IDASearcher *searcher = new IDASearcher(iddfs_heuristic, cornerSolver_goal, NULL);
  RubiksCube *cube = new RubiksCube;
  for (size_t ci = 0; ci < 8; ci++) {
    for (size_t goalOrient = 0; goalOrient < 3; goalOrient++) {
      for (size_t goalPos = 0; goalPos < 8; goalPos++) {
        uint8_t *pos_corner = new uint8_t[2];
        uint8_t goalCorner = (uint8_t)corners[ci] | goalOrient;
        pos_corner[0] = goalPos;
        pos_corner[1] = goalCorner;
        searcher->setGoalAux((void *)pos_corner);
        cube->resetCube();
        vector<RubiksCube::Rotation> moves = searcher->search(*cube);
        /*
        ci = C_WRB
        goalPos = C_YBO

        */
        manhattan_dist[goalPos][goalCorner] = moves.size();
        delete pos_corner;
      }
    }
  }
  delete cube;
}
int main() {
  fillManhattanArray();
  RubiksCube *cube = new RubiksCube;
  cube->scramble(9);
  cube->printCube();
  IDASearcher *searcher = new IDASearcher(iddfs_heuristic, solve_goal, NULL);
  IDASearcher *man_searcher = new IDASearcher(manhattan_heuristic, solve_goal, NULL);
  vector<RubiksCube::Rotation> moves = man_searcher->search(*cube);
  std::cout << "Manhattan searcher found solution " << moves.size() << "\n";
  vector<RubiksCube::Rotation> moves2 = searcher->search(*cube);
  std::cout << "IDDFS searcher found solution" << moves2.size() << "\n";

  // for (auto &move : moves) {
  //   std::cout << (int)move << "\n";
  // }
}

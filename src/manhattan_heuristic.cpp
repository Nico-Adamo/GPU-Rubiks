#include <bitset>
#include <cstring>
#include <iostream>

#include "search.hpp"

uint8_t manhattan_corners[8][255];
uint8_t manhattan_edges[12][255];

string cornerStrings[8] = {"C_WGR", "C_WRB", "C_WBO", "C_WGO", "C_YBR", "C_YRG", "C_YGO", "C_YBO"};
RubiksCube::CornerCubie corners[8] = {
    RubiksCube::CornerCubie::C_WGR, RubiksCube::CornerCubie::C_WRB, RubiksCube::CornerCubie::C_WBO,
    RubiksCube::CornerCubie::C_WGO, RubiksCube::CornerCubie::C_YBR, RubiksCube::CornerCubie::C_YRG,
    RubiksCube::CornerCubie::C_YGO, RubiksCube::CornerCubie::C_YBO};

RubiksCube::EdgeCubie edges[12] = {
    RubiksCube::EdgeCubie::E_WR, RubiksCube::EdgeCubie::E_WG, RubiksCube::EdgeCubie::E_WB,
    RubiksCube::EdgeCubie::E_WO, RubiksCube::EdgeCubie::E_YR, RubiksCube::EdgeCubie::E_YG,
    RubiksCube::EdgeCubie::E_YB, RubiksCube::EdgeCubie::E_YO, RubiksCube::EdgeCubie::E_RG,
    RubiksCube::EdgeCubie::E_RB, RubiksCube::EdgeCubie::E_OG, RubiksCube::EdgeCubie::E_OB};

uint8_t iddfs_heuristic(RubiksCube &cube) {
  return 0;
}

uint8_t manhattan_heuristic(RubiksCube &cube) {
  uint8_t corner_total = 0;
  uint8_t edge_total = 0;

  for (size_t i = 0; i < 8; i++) {
    corner_total += manhattan_corners[i][cube.getCorner(i)];
  }
  for (size_t i = 0; i < 12; i++) {
    edge_total += manhattan_edges[i][cube.getEdge(i)];
  }

  return max(corner_total / 4, edge_total / 4);
}

bool solve_goal(RubiksCube &cube, void *aux) {
  return cube.isSolved();
}

bool cornerSolver_goal(RubiksCube &cube, void *aux) {
  uint8_t pos = *((uint8_t *)aux);
  uint8_t corner = *(((uint8_t *)aux) + 1);
  return cube.getCorner(pos) == corner;
}

bool edgeSolver_goal(RubiksCube &cube, void *aux) {
  uint8_t pos = *((uint8_t *)aux);
  uint8_t edge = *(((uint8_t *)aux) + 1);
  return cube.getEdge(pos) == edge;
}

void fillManhattanCorners() {
  memset((void *)manhattan_corners, 0, 255 * 8 * sizeof(uint8_t));
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
        manhattan_corners[goalPos][goalCorner] = moves.size();
        delete pos_corner;
      }
    }
  }
  delete cube;
}
void fillManhattanEdges() {
  memset((void *)manhattan_edges, 0, 255 * 12 * sizeof(uint8_t));
  vector<RubiksCube::Rotation> moves;
  IDASearcher *searcher = new IDASearcher(iddfs_heuristic, edgeSolver_goal, NULL);
  RubiksCube *cube = new RubiksCube;
  for (size_t ei = 0; ei < 12; ei++) {
    for (size_t goalOrient = 0; goalOrient < 2; goalOrient++) {
      for (size_t goalPos = 0; goalPos < 12; goalPos++) {
        uint8_t *pos_edge = new uint8_t[2];
        uint8_t goalEdge = (uint8_t)edges[ei] | goalOrient;
        pos_edge[0] = goalPos;
        pos_edge[1] = goalEdge;
        searcher->setGoalAux((void *)pos_edge);
        cube->resetCube();
        vector<RubiksCube::Rotation> moves = searcher->search(*cube);
        manhattan_edges[goalPos][goalEdge] = moves.size();
        delete pos_edge;
      }
    }
  }
  delete cube;
}

int main() {
  fillManhattanCorners();
  fillManhattanEdges();

  RubiksCube *cube = new RubiksCube;
  cube->scramble(7);
  cube->printCube();
  IDASearcher *searcher = new IDASearcher(iddfs_heuristic, solve_goal, NULL);
  IDASearcher *man_searcher = new IDASearcher(manhattan_heuristic, solve_goal, NULL);
  vector<RubiksCube::Rotation> moves = man_searcher->search(*cube);
  std::cout << "Manhattan searcher found solution of length " << moves.size() << "\n";
  vector<RubiksCube::Rotation> moves2 = searcher->search(*cube);
  std::cout << "IDDFS searcher found solution of length " << moves2.size() << "\n";

  // for (auto &move : moves) {
  //   std::cout << (int)move << "\n";
  // }
}

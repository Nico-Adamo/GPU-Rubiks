#include <vector>

#include "rubiks.hpp"

using namespace std;

#define MAX_DEPTH 20

class IDASearcher {
 public:
  IDASearcher(uint8_t (*heuristic)(RubiksCube &));
  vector<RubiksCube::Rotation> search(RubiksCube &cube);

 private:
  uint8_t (*heuristic)(RubiksCube &);
};
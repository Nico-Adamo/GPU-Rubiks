#include <stdint.h>

using namespace std;

class RubiksCube {
  /**
   * Sides:
   *
   *    U
   *  L F R B
   *    D
   *

   * Indices:
   *
   *
   *              0  1  2
   *              7     3
   *              6  5  4
   *
   *   8  9 10   16 17 18   24 25 26   32 33 34
   *  15    11   23    19   31    27   39    35
   *  14 13 12   22 21 20   30 29 28   38 37 36
   *
   *             40 41 42
   *             47    43
   *             46 45 44
   *
   */

 public:
  enum class Face : uint8_t { UP, LEFT, FRONT, RIGHT, BACK, DOWN };

  enum class Color : uint8_t { WHITE, GREEN, RED, BLUE, ORANGE, YELLOW };
  enum class Rotation : uint8_t {
    U,       // Upper face clockwise
    U_PRIME, // Upper face counter-clockwise
    D,       // Down face clockwise
    D_PRIME, // Down face counter-clockwise
    L,       // Left face clockwise
    L_PRIME, // Left face counter-clockwise
    R,       // Right face clockwise
    R_PRIME, // Right face counter-clockwise
    F,       // Front face clockwise
    F_PRIME, // Front face counter-clockwise
    B,       // Back face clockwise
    B_PRIME, // Back face counter-clockwise
    LAST_MOVE
  };

  // Constructor
  RubiksCube();
  RubiksCube(const RubiksCube &cube);

  // Methods
  void resetCube();               // Resets the cube to its solved state
  void rotate(Rotation rotation); // Rotates the cube
  void printCube();               // Prints the current state of the cube
  void printCubeDebug();          // Prints the current state of the cube array
  bool isSolved();
  uint64_t getFace(Face f) const;
  void scramble(uint8_t depth);

  bool operator==(const RubiksCube &c) const;

 private:
  Color cube[48];
  void turnClockwise(Face f);
  void turnCounterClockwise(Face f);
  void updateSides(unsigned s0, unsigned s1, unsigned s2, unsigned s3, unsigned c0, unsigned c1,
                   unsigned c2, unsigned c3);
};

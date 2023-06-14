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
  enum class CornerCubie : uint8_t {
    C_WGR = 0b11100000,
    C_WRB = 0b10110000,
    C_WBO = 0b10011000,
    C_WGO = 0b11001000,
    C_YBR = 0b00110100,
    C_YRG = 0b01100100,
    C_YGO = 0b01001100,
    C_YBO = 0b00011100
  };

  enum class EdgeCubie : uint8_t {
    E_WR = 0b10100000,
    E_WG = 0b11000000,
    E_WB = 0b10010000,
    E_WO = 0b10001000,
    E_YR = 0b00100100,
    E_YG = 0b01000100,
    E_YB = 0b00010100,
    E_YO = 0b00001100,
    E_RG = 0b01100000,
    E_RB = 0b00110000,
    E_OG = 0b01001000,
    E_OB = 0b00011000
  };

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

  uint8_t getCorner(uint8_t pos);
  uint8_t getEdge(uint8_t pos);
  bool operator==(const RubiksCube &c) const;

 private:
  Color cube[48];
  void turnClockwise(Face f);
  void turnCounterClockwise(Face f);
  void updateSides(unsigned s0, unsigned s1, unsigned s2, unsigned s3, unsigned c0, unsigned c1,
                   unsigned c2, unsigned c3);
};
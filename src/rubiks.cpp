#include "rubiks.hpp"

#include <iostream>
#include <random>

static string FACES[6] = {"UP", "LEFT", "FRONT", "RIGHT", "BACK", "DOWN"};
static string COLORS[6] = {"W", "G", "R", "B", "O", "Y"};

std::random_device rd;
std::mt19937 eng(rd());

// Constructor
RubiksCube::RubiksCube() { resetCube(); }

RubiksCube::RubiksCube(const RubiksCube &cube_to_copy) {
  copy(begin(cube_to_copy.cube), end(cube_to_copy.cube), begin(cube));
}

void RubiksCube::resetCube() {
  int i = 0;
  for (uint8_t color = 0; color < 6; color++) {
    Color rubik_color = static_cast<Color>(color);
    do {
      cube[i] = rubik_color;
      i++;
    } while (i % 8 != 0);
  }
}

inline void RubiksCube::turnClockwise(Face f) {
  uint64_t face = *(uint64_t *)&cube[(unsigned)f * 8];
  asm volatile("rolq $16, %[face]" : [face] "+r"(face) :);
  *(uint64_t *)&cube[(unsigned)f * 8] = face;
}

inline void RubiksCube::turnCounterClockwise(Face f) {
  uint64_t face = *(uint64_t *)&cube[(unsigned)f * 8];
  asm volatile("rorq $16, %[face]" : [face] "+r"(face) :);
  *(uint64_t *)&cube[(unsigned)f * 8] = face;
}
/*
 Computes the permutation (s3, s0, s1, s2) on the cube indices
 Also computes (c3, c0, c1, c2)
*/
inline void RubiksCube::updateSides(unsigned s0, unsigned s1, unsigned s2, unsigned s3, unsigned c0,
                                    unsigned c1, unsigned c2, unsigned c3) {
  uint16_t s3_orig = *((uint16_t *)&cube[s3]);

  *((uint16_t *)&cube[s3]) = *((uint16_t *)&cube[s2]);
  *((uint16_t *)&cube[s2]) = *((uint16_t *)&cube[s1]);
  *((uint16_t *)&cube[s1]) = *((uint16_t *)&cube[s0]);
  *((uint16_t *)&cube[s0]) = s3_orig;

  Color c3_orig = cube[c3];
  cube[c3] = cube[c2];
  cube[c2] = cube[c1];
  cube[c1] = cube[c0];
  cube[c0] = c3_orig;
}

void RubiksCube::printCubeDebug() {
  for (size_t i = 0; i < 20; i++) {
    std::cout << unsigned(cube[i]) << " ";
  }
  std::cout << "\n";
}

void RubiksCube::printCube() {
  Color cube_arr[6][9];
  size_t spiral_idx[8] = {0, 1, 2, 5, 8, 7, 6, 3};
  for (size_t face = 0; face < 6; face++) {
    for (size_t i = 0; i < 8; i++) {
      cube_arr[face][spiral_idx[i]] = cube[face * 8 + i];
    }
    cube_arr[face][4] = static_cast<Color>(face);
  }

  for (size_t face = 0; face < 6; face++) {
    std::cout << FACES[face] << ":\n";
    for (size_t pos = 0; pos < 9; pos += 3) {
      std::cout << COLORS[(uint8_t)cube_arr[face][pos]] << " "
                << COLORS[(uint8_t)cube_arr[face][pos + 1]] << " "
                << COLORS[(uint8_t)cube_arr[face][pos + 2]] << "\n";
    }
  }
  std::cout << "\n";
}

void RubiksCube::scramble(uint8_t depth) {
  std::uniform_int_distribution<> distr(1, (int)RubiksCube::Rotation::LAST_MOVE - 1);
  for (uint8_t i = 0; i < depth; i++) {
    uint8_t randomMove = (uint8_t)distr(eng);
    this->rotate((RubiksCube::Rotation)randomMove);
  }
}

void RubiksCube::rotate(Rotation rotation) {
  switch (rotation) {
    case Rotation::U:
      turnClockwise(Face::UP);
      updateSides(32, 24, 16, 8, 34, 26, 18, 10);
      break;
    case Rotation::U_PRIME:
      turnCounterClockwise(Face::UP);
      updateSides(8, 16, 24, 32, 10, 18, 26, 34);
      break;
    case Rotation::D:
      turnClockwise(Face::DOWN);
      updateSides(12, 20, 28, 36, 14, 22, 30, 38);
      break;
    case Rotation::D_PRIME:
      turnCounterClockwise(Face::DOWN);
      updateSides(36, 28, 20, 12, 38, 30, 22, 14);
      break;
    case Rotation::L:
      turnClockwise(Face::LEFT);
      updateSides(6, 22, 46, 34, 0, 16, 40, 36);
      break;
    case Rotation::L_PRIME:
      turnCounterClockwise(Face::LEFT);
      updateSides(34, 46, 22, 6, 36, 40, 16, 0);
      break;
    case Rotation::R:
      turnClockwise(Face::RIGHT);
      updateSides(38, 42, 18, 2, 32, 44, 20, 4);
      break;
    case Rotation::R_PRIME:
      turnCounterClockwise(Face::RIGHT);
      updateSides(2, 18, 42, 38, 4, 20, 44, 32);
      break;
    case Rotation::F:
      turnClockwise(Face::FRONT);
      updateSides(30, 40, 10, 4, 24, 42, 12, 6);
      break;
    case Rotation::F_PRIME:
      turnCounterClockwise(Face::FRONT);
      updateSides(4, 10, 40, 30, 6, 12, 42, 24);
      break;
    case Rotation::B:
      turnClockwise(Face::BACK);
      updateSides(26, 0, 14, 44, 28, 2, 8, 46);
      break;
    case Rotation::B_PRIME:
      turnCounterClockwise(Face::BACK);
      updateSides(44, 14, 0, 26, 46, 8, 2, 28);
      break;
    default:
      break;
  }
}

uint64_t RubiksCube::getFace(Face f) const { return *(uint64_t *)&cube[(unsigned)f * 8]; }

bool RubiksCube::operator==(const RubiksCube &other) const {
  return getFace(Face::UP) == other.getFace(Face::UP) &&
         getFace(Face::LEFT) == other.getFace(Face::LEFT) &&
         getFace(Face::FRONT) == other.getFace(Face::FRONT) &&
         getFace(Face::RIGHT) == other.getFace(Face::RIGHT) &&
         getFace(Face::BACK) == other.getFace(Face::BACK) &&
         getFace(Face::DOWN) == other.getFace(Face::DOWN);
}

bool RubiksCube::isSolved() {
  return getFace(Face::UP) == 0x00000000 && getFace(Face::LEFT) == 0x101010101010101 &&
         getFace(Face::FRONT) == 0x202020202020202 && getFace(Face::RIGHT) == 0x303030303030303 &&
         getFace(Face::BACK) == 0x404040404040404 && getFace(Face::DOWN) == 0x505050505050505;
}

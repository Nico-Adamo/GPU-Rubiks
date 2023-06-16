#pragma once

#include <vector>

#include "rubiks.hpp"
using namespace std;

#define MAX_DEPTH 20

uint8_t gpu_search(RubiksCube &cube);
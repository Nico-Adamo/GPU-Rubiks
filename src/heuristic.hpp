#pragma once

#include "search.hpp"

extern uint8_t manhattan_corners[8][255];
extern uint8_t manhattan_edges[12][255];

void manhattan_init();

uint8_t manhattan_heuristic(RubiksCube &cube);
uint8_t iddfs_heuristic(RubiksCube &cube);
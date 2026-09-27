#pragma once

#include <string>
#include <vector>

bool readMatrixFromFile(int n, const std::string& filename, std::vector<double>& matrix, std::string& error);
bool initializeMatrix(int n, int k, const std::string& filename, std::vector<double>& matrix, std::string& error);

#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <random>
#include <vector>

#include "exp.h"

namespace fs = std::filesystem;

int main(int argc, char *argv[]) {
  try {
    if (argc > 3) {
      return Experiment::load_from_args(argv);
    }

    return Experiment::load_from_file();
  } catch (const std::exception &e) {
    std::cerr << e.what() << std::endl;

    return 1;
  }
}

#pragma once

#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <random>
#include <vector>

#define DATASETS_RESERVE 100

namespace fs = std::filesystem;
using duration = std::chrono::duration<float>;

namespace Kmeans {

enum class OutputType : uint8_t {
  Iteration,
  AllIterations,
  Overall,
  IterationCount,
  Init
};

struct Dataset {
  const fs::path image;
  const uint16_t repeat;
  const std::vector<uint32_t> ks;

  Dataset(const fs::path &_image, const uint16_t _repeat,
          const std::vector<uint32_t> &_ks)
      : image(_image), repeat(_repeat), ks(_ks) {}
};

struct Pixel {
  int32_t r, g, b;
};

struct PixelCoord : Pixel {
  uint32_t x, y;
};

struct Result {
  const duration init_in_seconds, iterations_in_seconds;
  const uint32_t iterations_count, max_iterations;
  const std::unique_ptr<std::vector<Pixel>> means_ptr;
  const std::unique_ptr<std::vector<size_t>> classes_ptr;

  constexpr duration iteration() const {
    return iterations_in_seconds / static_cast<long double>(iterations_count);
  }

  constexpr duration overall() const {
    return init_in_seconds + iterations_in_seconds;
  }

  constexpr bool max_interations_reached() const {
    return iterations_count == max_iterations;
  }

  constexpr duration from_output_type(const OutputType type) const {
    switch (type) {
    case OutputType::Init:
      return init_in_seconds;
    case OutputType::Iteration:
      return iteration();
    case OutputType::AllIterations:
      return iterations_in_seconds;
    default:
      return overall();
    }
  }

  inline const std::vector<Pixel> &means() const { return *means_ptr; }
  inline const std::vector<size_t> &classes() const { return *classes_ptr; }
};

struct ResultMean {
private:
  long double n;
  long double init = 0.0f, iteration = 0.0f, iterations_count = 0.0f;

public:
  ResultMean(const uint32_t _n) : n(_n) {}

  ResultMean &operator+=(const Result &result) {
    init += result.init_in_seconds.count() / n;
    iteration += result.iteration().count() / n;
    iterations_count += result.iterations_count / n;

    return *this;
  }

  constexpr long double from_output_type(const OutputType type) const {
    switch (type) {
    case OutputType::Init:
      return init;
    case OutputType::Iteration:
      return iteration;
    case OutputType::IterationCount:
      return iterations_count;
    default:
      throw new std::out_of_range("not supported");
    }
  }
};

Kmeans::Result kmeans(const std::vector<Kmeans::PixelCoord> &dataset,
                      const size_t N, const uint32_t K,
                      const uint32_t max_iterations = 1000);
constexpr const char *output_type_to_string(const Kmeans::OutputType type);
} // namespace Kmeans
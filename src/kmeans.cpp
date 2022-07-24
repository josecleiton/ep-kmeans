#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <random>
#include <vector>

#include "kmeans.h"

constexpr const char *output_type_to_string(const Kmeans::OutputType type) {
  switch (type) {
  case Kmeans::OutputType::Init:
    return "init";
  case Kmeans::OutputType::Iteration:
    return "iteration";
  case Kmeans::OutputType::AllIterations:
    return "all_iterations";
  case Kmeans::OutputType::IterationCount:
    return "iteration_count";
  default:
    return "overall";
  }
}

inline long double d(const Kmeans::Pixel &p, const Kmeans::Pixel &q) {
  const auto r = static_cast<long double>(p.r) - q.r; // (2, 1, 0)
  const auto g = static_cast<long double>(p.g) - q.g; // (2, 1, 0)
  const auto b = static_cast<long double>(p.b) - q.b; // (2, 1, 0)

  return std::sqrt(r * r + g * g + b * b);
  // 3* (2, 1, 0) + (5, 5, 0) + (2, 1, 0) = (13, 9, 0)
}

// ANALISE QUANTITATIVA DA FUNÇÃO kmeans
// (4, 0, 1) + K * (4, 1, 1) +
// (3, 0, 1) + N * ((2, 1, 2)) +
// (5, 0, 0) + K * ((1, 0, 0)) +
// (0, 0, 1) + X * (
//    (2, 2, 2) + (
//       (1, 0, 1) + N * ((1, 1, 1) + (4, 0, 1) +  (1, 0, 1) + K * (
//          (1, 1, 1) + (16, 9, 1)
//      )) +
//       (1, 0, 1) + K * ((1, 1, 1) + (10, 3, 1) + (1, 0, 1) + N * (
//          (1, 1, 1) + (7, 4, 1)
//      ))
//    )
// )
//
// JUNTA OS TERMOS EM COMUM
//
// (12, 0, 3) + K * (5, 1, 1) + N * (2, 1, 2) + X * (
//    (4, 2, 4) + N * ((6, 1, 3) + K * (17, 10, 2)) +
//    K * ((12, 4, 3) + N * (8, 5, 2))
// )
//
// (12, 0, 3) + K * (5, 1, 1) + N * (2, 1, 2) + X * (
//    (4, 2, 4) + N * (6, 1, 3) + (N * K) * (17, 10, 2) +
//    K * (12, 4, 3) + (N * K) * (8, 5, 2)
// )
//
// (12, 0, 3) + K * (5, 1, 1) + N * (2, 1, 2) + X * (
//    (4, 2, 4) + N * (6, 1 ,3) + K * (12, 4, 3) + (N * K) * (25, 15, 4)
// )
//
// Separando (A, O, C)
// A = 12 + 5K + 2N + X (4 + 6N + 12K + 25NK)
// O = K + N + X (2 + N + 4K + 15NK)
// C = 3 + K + 2N + X (4 + 3N + 3K + 4NK)
//
//  INIT
// A = 12 + 5K + 2N
// O = K + N
// C = 3 + K + 2N
//
//  ITERATION
// A = 4 + 6 + 12K + 25NK
// O = 2 + N + 4K + 15NK
// C = 4 + 3N + 3K + 4NK
//
// Utilizar a aula 11 (1h01min) para construir a tabela e ter as normas L1 e L2

Kmeans::Result kmeans(const std::vector<Kmeans::PixelCoord> &dataset,
                      const size_t N, const uint32_t K,
                      const uint32_t max_iterations = 1000) {

  std::random_device rdev;
  std::mt19937 eng{rdev()};
  std::uniform_int_distribution<int> dist(0, N - 1);

  const auto init_time_start = std::chrono::high_resolution_clock::now();

  auto means_ptr =
      std::make_unique<std::vector<Kmeans::Pixel>>(K); // (K + 2, 0, 0)
  auto &means = *means_ptr;                            // (1, 0, 0)
  for (uint32_t k = 0; k < K; ++k) {
    // g12(1, 0, 1); gr2(1, 1, 1); e2(2, 0, 0)
    means[k] = dataset[dist(eng)];
  }

  auto classes_ptr = std::make_unique<std::vector<size_t>>(
      N, std::numeric_limits<size_t>::max()); // (2, 0, 1) + N * (2, 1, 2)
  auto &classes = *classes_ptr;               // (1, 0, 0)

  long double distance, minimum;            // (2, 0, 0)
  uint32_t x = 0;                           // (1, 0, 0)
  bool changed;                             // (1, 0, 0)
  std::vector<uint32_t> cluster_counter(K); // (K, 0, 0)
  size_t new_class = 0;                     // (1, 0, 0)

  const auto init_time_end = std::chrono::high_resolution_clock::now();

  const auto iterations_time_start = std::chrono::high_resolution_clock::now();
  for (; x < max_iterations; ++x) {
    // g13(0, 0, 1); gr3(1, 1, 1);
    // ex3 = (1, 1, 1) + (gr4 + ex4) + (gr6 + ex6)
    changed = false; // (1, 0, 0)

    for (size_t i = 0; i < N; ++i) {
      // g14(1, 0, 1); gr4(1, 1, 1); ex4 = (4, 0, 1) + N * (gr5 + ex5)
      minimum = std::numeric_limits<long double>::max(); // (1, 0, 0)
      new_class = classes[i];                            // (1, 0, 0)

      for (uint32_t k = 0; k < K; ++k) {
        // g15(1, 0, 1); gr5(1, 1, 1); ex5 = (16, 9, 1)
        distance =                   // (1, 0 ,0)
            d(dataset[i], means[k]); // inline function: (13, 9, 0)

        if (distance < minimum) { // (0, 0, 1) + 2*(1, 0, 0) = (2, 0, 1)
          minimum = distance;     // (1, 0, 0)
          new_class = k;          // (1, 0, 0)
        }
      }

      if (new_class != classes[i]) { // (0, 0, 1) + 2 * (1, 0, 0) = (2, 0, 1)
        changed = true;
        classes[i] = new_class;
      }
    }

    if (!changed) { // (0, 1, 1)
      break;
    }

    for (uint32_t k = 0; k < K; ++k) {
      // g16(1, 0, 1); gr6(1, 1, 1); ex6 = (4, 0, 0) + (6, 3, 1) = (10, 3, 1)
      means[k].r = means[k].g = means[k].b = 0; // (3, 0, 0)
      cluster_counter[k] = 0;                   // (1, 0, 0)

      for (size_t i = 0; i < N; ++i) {
        // g17(1, 0, 1); gr7(1, 1, 1); ex7 = (7, 4, 1)
        if (classes[i] == k) {        // (0, 0, 1) + 3 * (2, 1, 0) + (1, 1, 0)
          means[k].r += dataset[i].r; // (2, 1, 0)
          means[k].g += dataset[i].g; // (2, 1, 0)
          means[k].b += dataset[i].b; // (2, 1, 0)
          ++cluster_counter[k];       // (1, 1, 0)
        }
      }

      if (cluster_counter[k]) { // (0, 0, 1) + 3 * (2, 1, 0) = (6, 3, 1)
        means[k].r /= cluster_counter[k]; // (2, 1, 0)
        means[k].g /= cluster_counter[k]; // (2, 1, 0)
        means[k].b /= cluster_counter[k]; // (2, 1, 0)
      }
    }
  }

  const auto iterations_time_end = std::chrono::high_resolution_clock::now();

  if (x == max_iterations) {
    std::clog << "clustering finished due to MAX_ITERATIONS reached\n";
  }

  return {init_time_end - init_time_start,
          iterations_time_end - iterations_time_start,
          x,
          max_iterations,
          std::move(means_ptr),
          std::move(classes_ptr)};
}

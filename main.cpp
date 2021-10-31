#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <memory>
#include <random>
#include <vector>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#define MAX 100000
#define CHANNELS 3

using duration = std::chrono::duration<float>;

struct Dataset {
  const std::string image;
  const std::vector<uint32_t> ks;
  const uint16_t repeat;
};

struct Pixel {
  uint8_t r, g, b;
};

struct PixelCoord : Pixel {
  size_t x, y;
};

struct KMeansResult {
  const duration init_in_seconds, iterations_in_seconds;
  const uint32_t iterations_count;
  const std::unique_ptr<std::vector<Pixel>> means_ptr;
  const std::unique_ptr<std::vector<size_t>> belongs_ptr;

  constexpr duration iteration() const {
    return iterations_in_seconds / iterations_count;
  }

  constexpr duration overall() const {
    return init_in_seconds + iterations_in_seconds;
  }

  const std::vector<Pixel> &means() const { return *means_ptr; }
  const std::vector<size_t> &belongs() const { return *belongs_ptr; }
};

inline int d(const Pixel &c, const Pixel &p) {
  return sqrt(pow(c.r - p.r, 2) + pow(c.g - p.g, 2) + pow(c.b - p.b, 2));
  // 3*(2, 1, 0) + 2*(1, 1, 0) + (1, 0, 0) = (9, 5, 0)
}

// ANALISE QUANTITATIVA DA FUNÇÃO kmeans
// (N, 0, 0) + ((1, 0, 1) + N * (2, 1, 1)) +
// (K, 0, 0) + ((1, 0, 1) + K * (4, 1, 1)) +
// (6, 0, 0) + (K, 0, 0) + (0, 0, 1) +
// X * (
//  (1, 0, 1) + ((1, 0, 1) + K * (16, 8, 2)) +
//  ((1, 0, 1) + K* ((7, 3, 1) + (1, 0, 1) + N*(N-1)/2 * (4, 4, 1))
// )
// JUNTANDO TERMOS COM FATORES IGUAIS:
// (8, 0, 3) +
// N * (3, 1, 1) +
// K * (6, 1, 1) +
// X * (
//  (3, 0, 3) +
//  N * ( (1, 0, 0) + K * (16, 8, 2) ) +
//  K * ( (9, 4, 3) + N * (N - 1) / 2 * (4, 4, 1) )
// )

KMeansResult kmeans(const std::vector<PixelCoord> &dataset, const size_t N,
                    const uint32_t K, const uint32_t max_iterations = 1000) {

  std::random_device rdev;
  std::mt19937 eng{rdev()};
  std::uniform_int_distribution<int> dist(0, N);

  const auto init_time_start = std::chrono::high_resolution_clock::now();

  auto belongs_ptr = std::make_unique<std::vector<size_t>>(N); // (N, 0, 0)
  auto &belongs = *belongs_ptr;                                // (1, 0, 0)
  for (size_t i = 0; i < N; ++i) { // g11(1, 0, 1); gr1(1, 1, 1);
                                   // e1(1, 0, 0)
    belongs[i] = K + 1;
  }

  auto means_ptr = std::make_unique<std::vector<Pixel>>(K); // (K, 0, 0)
  auto &means = *means_ptr;                                 // (1, 0, 0)
  for (uint32_t k = 0; k < K; ++k) {
    // g12(1, 0, 1); gr2(1, 1, 1); e2(3, 0, 0)
    means[k] = dataset[dist(eng)];
  }

  const auto init_time_end = std::chrono::high_resolution_clock::now();

  const auto iterations_time_start = std::chrono::high_resolution_clock::now();

  size_t distance, minimum;            // (2, 0, 0)
  uint32_t x = 0;                      // (1, 0, 0)
  long long changed;                   // (1, 0, 0)
  std::vector<int> cluster_counter(K); // (K, 0, 0)
  for (; x < max_iterations; ++x) {
    // g13(0, 0, 1); gr3(1, 1, 1);
    // ex3 = (1, 0, 1) + (K, 0, 0) + bloco4 + bloco6
    changed = 0; // (1, 0, 0)

    for (size_t i = 0; i < N; ++i) {
      // g14(1, 0, 1); gr4(1, 1, 1); ex4 = (1, 0, 0) + N * (bloco5)
      minimum = std::numeric_limits<size_t>::max();

      for (uint32_t k = 0; k < K; ++k) {
        // g15(1, 0, 1); gr5(1, 1, 1); ex5 = (15, 7, 1)
        distance =                   // (1, 0 ,0)
            d(dataset[i], means[k]); // inline function: (9, 5, 0)

        if (distance < minimum) {    // total = (5, 2, 1)
          minimum = distance;        // (1, 0, 0)
          changed += belongs[i] - k; // (3, 2, 0)
          belongs[i] = k;            // (1, 0, 0)
        }
      }
    }

    if (changed == 0) { // (0, 0, 1)
      break;
    }

    for (uint32_t k = 0; k < K; ++k) {
      // g16(1, 0, 1); gr6(1, 1, 1); ex6 = (4, 0, 0) + (3, 3, 1)
      means[k].r = means[k].g = means[k].b = 0; // (3, 0, 0)
      cluster_counter[k] = 0;                   // (1, 0, 0)

      for (size_t i = 0; i < N; ++i) {
        // g17(1, 0, 1); gr7(1, 1, 1); ex7 = (4, 4, 1) * N(N -1)/2
        if (belongs[i] == k) {        // (4, 4, 1)
          means[k].r += dataset[i].r; // (1, 1, 0)
          means[k].g += dataset[i].g; // (1, 1, 0)
          means[k].b += dataset[i].b; // (1, 1, 0)
          ++cluster_counter[k];       // (1, 1, 0)

          break;
        }
      }

      if (cluster_counter[k]) {           // (3, 3, 1)
        means[k].r /= cluster_counter[k]; // (1, 1, 0)
        means[k].g /= cluster_counter[k]; // (1, 1, 0)
        means[k].b /= cluster_counter[k]; // (1, 1, 0)
      }
    }
  }

  const auto iterations_time_end = std::chrono::high_resolution_clock::now();

  if (x == max_iterations) {
    std::clog << "clustering finished due to MAX_ITERATIONS reached\n";
  }

  return {init_time_end - init_time_start,
          iterations_time_end - iterations_time_start, x, std::move(means_ptr),
          std::move(belongs_ptr)};
}

// IGNORE: MAGICA
std::unique_ptr<std::vector<PixelCoord>>
load_dataset(const std::string &file_location) {
  int width, height, bpp;
  uint8_t *const rgb_image =
      stbi_load(file_location.c_str(), &width, &height, &bpp, CHANNELS);

  if (rgb_image == nullptr) {
    std::cerr << "error loading image " << '\'' << file_location << "'\n"
              << "reason: " << stbi_failure_reason() << '\n';
    exit(1);
  }

  auto result_ptr = std::make_unique<std::vector<PixelCoord>>(width * height);
  auto &result = *result_ptr;

  int r_index, dest;
  for (int j = 0; j < height; j++) {
    for (int i = 0; i < width; i++) {
      r_index = CHANNELS * i + CHANNELS * width * j;
      dest = j * height + i;

      result[dest].r = rgb_image[r_index];
      result[dest].g = rgb_image[1 + r_index];
      result[dest].b = rgb_image[2 + r_index];
      result[dest].x = i;
      result[dest].y = j;
    }
  }

  stbi_image_free(rgb_image);

  return result_ptr;
}

void write_result_csv(std::ofstream &file, const KMeansResult &result,
                      const uint16_t i) {
  if (i != 0) {
    file << ",\n";
  } else {
    file << "time,\n";
  }

  file << result.overall().count();
}

int exp(const std::vector<Dataset> &datasets) {

  size_t dataset_id = 1;
  for (const auto &dataset : datasets) {
    const auto &pixels_ptr = load_dataset(dataset.image);
    const auto n = pixels_ptr->size();

    std::clog << "image: " << dataset.image << '\n'
              << "pixels count: " << n << '\n';

    for (const auto k : dataset.ks) {
      std::ofstream file("result_" + std::to_string(dataset_id) + ".csv",
                         std::fstream::out);

      for (uint16_t i = 0; i < dataset.repeat; i++) {
        if (n < k) {
          std::cerr << "number of clusters must be less than " << n << '\n';
          return 1;
        }

        if (k > 1) {
          std::clog << '\n';
        }

        std::clog << "kmeans begin (" << i + 1 << ")\n";

        const auto &result = kmeans(*pixels_ptr, n, k);
        // results.emplace_back(result);

        assert(k == result.means().size());
        assert(n == result.belongs().size());

        std::clog << "kmeans clusters: " << result.means().size() << '\n'
                  << "iterations count: " << result.iterations_count << '\n'
                  << "init time: " << result.init_in_seconds.count() << "s\n"
                  << "overall iterations time: "
                  << result.iterations_in_seconds.count() << "s\n"
                  << "iteration mean time: " << result.iteration().count()
                  << "s\n"
                  << "kmeans finished\n";

        write_result_csv(file, result, i);
      }
    }

    dataset_id++;
  }

  return 0;
}

int main(int argc, char *argv[]) {
  if (argc > 0) {
    std::vector<Dataset> datasets;
    datasets.reserve(100);

    std::ifstream file("experimental", std::fstream::in);
    std::string filename;
    uint16_t k;

    while (file >> filename >> k) {
      datasets.push_back({"images/" + filename, {k}, 100});

      std::ifstream check_file(datasets.back().image);
      if (!check_file.is_open()) {
        std::cerr << "file " << datasets.back().image << " not found\n";
        exit(1);
      }
      check_file.close();
    }

    file.close();

    std::clog << "read " << datasets.size() << " photos\n";

    return exp(datasets);
  }

  return exp({{"images/image_teste_segementacao_3_classes.jpg", {15}, 100}});
}

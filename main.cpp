#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <random>
#include <vector>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#define IMAGE_CHANNELS 3
#define DATASETS_RESERVE 100

using duration = std::chrono::duration<float>;
namespace fs = std::filesystem;

enum class KMeansOutputType : uint8_t { Iteration, Overall };

struct Dataset {
  const fs::path &image;
  const std::unique_ptr<std::vector<uint32_t>> &ks_ptr;
  const uint16_t repeat;

  inline const std::vector<uint32_t> &ks() const { return *ks_ptr; }
};

struct Pixel {
  int32_t r, g, b;
};

struct PixelCoord : Pixel {
  uint32_t x, y;
};

struct KMeansResult {
  const duration init_in_seconds, iterations_in_seconds;
  const uint32_t iterations_count, max_iterations;
  const std::unique_ptr<std::vector<Pixel>> means_ptr;
  const std::unique_ptr<std::vector<size_t>> classes_ptr;

  constexpr duration iteration() const {
    return iterations_in_seconds / iterations_count;
  }

  constexpr duration overall() const {
    return init_in_seconds + iterations_in_seconds;
  }

  constexpr bool max_interations_reached() const {
    return iterations_count == max_iterations;
  }

  inline const std::vector<Pixel> &means() const { return *means_ptr; }
  inline const std::vector<size_t> &classes() const { return *classes_ptr; }
};

inline long double d(const Pixel &p, const Pixel &q) {
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
// Utilizar a aula 11 (1h01min) para construir a tabela e ter as normas L1 e L2

KMeansResult kmeans(const std::vector<PixelCoord> &dataset, const size_t N,
                    const uint32_t K, const uint32_t max_iterations = 1000) {

  std::random_device rdev;
  std::mt19937 eng{rdev()};
  std::uniform_int_distribution<int> dist(0, N - 1);

  const auto init_time_start = std::chrono::high_resolution_clock::now();

  auto means_ptr = std::make_unique<std::vector<Pixel>>(); // (2, 0, 0)
  auto &means = *means_ptr;                                // (1, 0, 0)
  means.reserve(K);                                        // (K, 0, 0)
  for (uint32_t k = 0; k < K; ++k) {
    // g12(1, 0, 1); gr2(1, 1, 1); e2(2, 0, 0)
    means[k] = dataset[dist(eng)];
  }

  const auto init_time_end = std::chrono::high_resolution_clock::now();

  const auto iterations_time_start = std::chrono::high_resolution_clock::now();

  auto classes_ptr = std::make_unique<std::vector<size_t>>(
      N, std::numeric_limits<size_t>::max()); // (2, 0, 1) + N * (2, 1, 2)
  auto &classes = *classes_ptr;               // (1, 0, 0)

  long double distance, minimum;            // (2, 0, 0)
  uint32_t x = 0;                           // (1, 0, 0)
  bool changed;                             // (1, 0, 0)
  std::vector<uint32_t> cluster_counter(K); // (K, 0, 0)
  size_t new_class = 0;                     // (1, 0, 0)
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

// IGNORE: MAGICA
std::unique_ptr<std::vector<PixelCoord>>
load_dataset(const fs::path &file_location) {
  int w, h, bpp;
  uint8_t *const rgb_image =
      stbi_load(file_location.string().c_str(), &w, &h, &bpp, IMAGE_CHANNELS);

  if (rgb_image == nullptr) {
    throw std::domain_error("error loading image '" + file_location.string() +
                            "'\nreason: " + stbi_failure_reason());
  }

  const auto height = static_cast<uint32_t>(h);
  const auto width = static_cast<uint32_t>(w);
  auto result_ptr = std::make_unique<std::vector<PixelCoord>>(width * height);
  auto &result = *result_ptr;

  size_t dest, src_index = 0;
  for (uint32_t i = 0; i < height; ++i) {
    for (uint32_t j = 0; j < width; ++j) {
      dest = static_cast<size_t>(i) * width + j;

      result[dest].r = rgb_image[src_index++];
      result[dest].g = rgb_image[src_index++];
      result[dest].b = rgb_image[src_index++];
      result[dest].x = j;
      result[dest].y = i;
    }
  }

  stbi_image_free(rgb_image);

  return result_ptr;
}

void write_result_csv(std::ofstream &file, const KMeansResult &result,
                      const uint16_t i, const KMeansOutputType type) {
  if (i != 0) {
    file << ",\n";
  } else {
    file << "time,\n";
  }

  if (type == KMeansOutputType::Iteration) {
    file << result.iteration().count();
  } else {
    file << result.overall().count();
  }
}

int exp(const std::vector<Dataset> &datasets,
        const KMeansOutputType outputType) {

  size_t dataset_id = 1;
  for (const auto &dataset : datasets) {
    const auto &pixels_ptr = load_dataset(dataset.image);
    const auto n = pixels_ptr->size();

    std::clog << "image: " << dataset.image << '\n'
              << "pixels count: " << n << '\n';

    for (const auto k : dataset.ks()) {
      const auto filepath = "output" / fs::path("result_") +=
          fs::path(dataset.image).stem() += "_" + std::to_string(k) += ".csv";

      std::ofstream file(filepath, std::fstream::out);
      if (!file.is_open()) {
        throw std::domain_error("output file not opened: '" +
                                filepath.string() + "'");
      }

      for (uint16_t i = 0; i < dataset.repeat; ++i) {
        if (n < k) {
          throw std::domain_error("number of clusters must be less than " +
                                  std::to_string(n));
        }

        if (k > 1) {
          std::clog << '\n';
        }

        std::clog << "kmeans begin (" << i + 1 << ")\n";

        const auto &result = kmeans(*pixels_ptr, n, k);

        assert(k == result.means().size());
        assert(n == result.classes().size());

        std::clog << "kmeans clusters: " << result.means().size() << '\n'
                  << "iterations count: " << result.iterations_count << '\n'
                  << "init time: " << result.init_in_seconds.count() << "s\n"
                  << "overall iterations time: "
                  << result.iterations_in_seconds.count() << "s\n"
                  << "iteration mean time: " << result.iteration().count()
                  << "s\n"
                  << "means colors: ";
        for (const auto &mean : result.means()) {
          std::clog << "(" << mean.r << ", " << mean.g << ", " << mean.b
                    << ") ";
        }

        std::clog << "\nkmeans finished\n";

        write_result_csv(file, result, i, outputType);
      }
    }

    ++dataset_id;
  }

  return 0;
}

int main(int argc, char *argv[]) {
  try {
    if (argc > 3) {
      return exp({{fs::path(argv[1]),
                   std::make_unique<std::vector<uint32_t>>(
                       1, static_cast<uint32_t>(std::atoi(argv[2]))),
                   static_cast<uint16_t>(std::atoi(argv[3]))}},
                 KMeansOutputType::Iteration);
    }

    std::vector<Dataset> datasets;
    datasets.reserve(DATASETS_RESERVE);

    std::ifstream file("experimental", std::fstream::in);
    std::string filename;
    uint16_t nk;
    uint32_t k;

    while (file >> filename >> nk) {
      auto ks = std::make_unique<std::vector<uint32_t>>();
      ks->reserve(nk);

      for (uint16_t i = 0; i < nk; i++) {
        file >> k;
        ks->emplace_back(k);
      }

      datasets.push_back({"images" / fs::path(filename), std::move(ks), 100});

      if (!fs::exists(datasets.back().image)) {
        throw std::domain_error("file '" + datasets.back().image.string() +
                                "' not found");
      }
    }

    file.close();

    std::clog << "read " << datasets.size() << " photos\n";

    return exp(datasets, KMeansOutputType::Overall);
  } catch (const std::exception &e) {
    std::cerr << e.what() << '\n';

    return 1;
  }
}

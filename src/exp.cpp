#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <random>
#include <vector>

#include "exp.h"

#include "kmeans.h"

#define STB_IMAGE_IMPLEMENTATION
#include "../lib/stb_image.h"

#define IMAGE_CHANNELS 3
#define DEFAULT_REPEATITION 20

namespace fs = std::filesystem;

void write_result_csv(std::ofstream &file, const Kmeans::Result &result,
                      const uint16_t i,
                      const std::vector<Kmeans::OutputType> types) {
  if (i == 1) {
    for (size_t j = 0; j < types.size(); j++) {
      if (j != 0)
        file << ',';
      file << Kmeans::output_type_to_string(types[j]);
    }
    file << '\n';
  }

  for (size_t j = 0; j < types.size(); j++) {
    if (j != 0)
      file << ',';
    if (types[j] != Kmeans::OutputType::IterationCount) {
      file << result.from_output_type(types[j]).count();
    } else {
      file << result.iterations_count;
    }
  }
  file << '\n';
}

void write_result_csv(std::ofstream &file,
                      const Kmeans::ResultMean &result_mean,
                      const std::vector<Kmeans::OutputType> types) {
  file << '\n';
  for (size_t i = 0; i < types.size(); i++) {
    if (i != 0)
      file << ',';
    file << result_mean.from_output_type(types[i]);
  }
}

std::unique_ptr<std::vector<Kmeans::PixelCoord>>
load_dataset(const fs::path &file_location) {
  int w, h, bpp;
  uint8_t *const rgb_image =
      stbi_load(file_location.string().c_str(), &w, &h, &bpp, IMAGE_CHANNELS);

  if (!rgb_image) {
    throw std::domain_error(std::string("error loading image: ") +
                            stbi_failure_reason() + " " +
                            file_location.string());
  }

  const auto height = static_cast<uint32_t>(h);
  const auto width = static_cast<uint32_t>(w);
  auto result_ptr =
      std::make_unique<std::vector<Kmeans::PixelCoord>>(width * height);
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

int exp(const std::vector<Kmeans::Dataset> &datasets,
        const std::vector<Kmeans::OutputType> &outputTypes) {

  for (const auto &dataset : datasets) {

    const auto pixels_ptr = load_dataset(dataset.image);
    const auto n = pixels_ptr->size();

    std::clog << "image: " << dataset.image << '\n'
              << "pixels count: " << n << '\n'
              << "ks: " << dataset.ks.size() << std::endl;

    for (const auto k : dataset.ks) {
      const auto filepath = "output" / fs::path("result_") +=
          fs::path(dataset.image).stem() += "_" + std::to_string(k) += ".csv";
      Kmeans::ResultMean result_mean(dataset.repeat);

      std::ofstream file(filepath, std::fstream::out);
      if (!file.is_open()) {
        throw std::domain_error("output file not opened: '" +
                                filepath.string() + "'");
      }

      for (uint32_t count = 1;
           count < static_cast<uint32_t>(dataset.repeat) + 1; ++count) {
        if (n < k) {
          throw std::domain_error("number of clusters must be less than " +
                                  std::to_string(n));
        }

        if (k > 1) {
          std::clog << '\n';
        }

        std::clog << "kmeans begin (" << count << ")\n";

        const auto &result = kmeans(*pixels_ptr, n, k);

        assert(k == result.means().size());
        assert(n == result.classes().size());

        std::clog << "clusters: " << result.means().size() << '\n'
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

        std::clog << '\n' << std::endl;

        write_result_csv(file, result, count, outputTypes);
        result_mean += result;
      }
      write_result_csv(file, result_mean, outputTypes);
    }
  }

  return 0;
}

int load_from_args(char *argv[]) {
  const std::vector<Kmeans::Dataset> datasets = {Kmeans::Dataset(
      fs::path(argv[1]), static_cast<uint32_t>(std::atoi(argv[3])),
      {static_cast<uint32_t>(std::atoi(argv[2]))})};
  return exp(datasets,
             {Kmeans::OutputType::Init, Kmeans::OutputType::Iteration});
}

int load_from_file() {
  std::vector<Kmeans::Dataset> datasets;
  datasets.reserve(DATASETS_RESERVE);

  std::ifstream file("experimental", std::fstream::in);
  std::string filename;
  uint16_t nk;

  while (file >> filename >> nk) {
    std::vector<uint32_t> ks(nk);

    for (uint16_t i = 0; i < nk; i++) {
      file >> ks[i];
    }

    datasets.emplace_back("images" / fs::path(filename), DEFAULT_REPEATITION,
                          ks);
    const auto &dataset = datasets.back();

    if (!fs::exists(dataset.image)) {
      throw std::domain_error("file " + dataset.image.string() + " not found");
    }
  }

  file.close();

  std::clog << "read " << datasets.size() << " photos\n";

  return exp(datasets, {Kmeans::OutputType::Init, Kmeans::OutputType::Iteration,
                        Kmeans::OutputType::IterationCount});
}

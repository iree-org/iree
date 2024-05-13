// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "experimental/hal_executable_library_call_hooks/stats.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>

float sum(const std::vector<float> &v) {
  float s = 0;
  for (float x : v) {
    s += x;
  }
  return s;
}

float mean(const std::vector<float> &v) { return sum(v) / v.size(); }

float covariance(const std::vector<float> &u, const std::vector<float> &v) {
  assert(u.size() == v.size());
  float mean_u = mean(u);
  float mean_v = mean(v);
  float s = 0;
  for (size_t i = 0; i < u.size(); ++i) {
    s += (u[i] - mean_u) * (v[i] - mean_v);
  }
  return s / u.size();
}

float variance(const std::vector<float> &v) { return covariance(v, v); }

float correlation(const std::vector<float> &u, const std::vector<float> &v) {
  return covariance(u, v) / std::sqrt(variance(u) * variance(v));
}

void splitIntoBuckets(const std::vector<float> &v, int bucket_count,
                      std::vector<float> *bucket_means,
                      std::vector<int> *bucket_indices) {
  bucket_means->resize(bucket_count);
  bucket_indices->resize(v.size());
  std::vector<float> sorted(v);
  std::sort(sorted.begin(), sorted.end());
  std::vector<int> bucket_delim_sorted_indices(bucket_count + 1);
  for (int i = 0; i < bucket_count; ++i) {
    bucket_delim_sorted_indices[i] =
        std::min<int>(sorted.size() - 1, sorted.size() * i / bucket_count);
  }
  bucket_delim_sorted_indices[bucket_count] = sorted.size();
  for (int i = 0; i < bucket_count; ++i) {
    (*bucket_means)[i] = mean(std::vector<float>(
        sorted.begin() + bucket_delim_sorted_indices[i],
        sorted.begin() + bucket_delim_sorted_indices[i + 1]));
  }
  for (size_t i = 0; i < v.size(); ++i) {
    float val = v[i];
    int bucket_index = 0;
    while (bucket_index < bucket_count - 1 &&
           val > sorted[bucket_delim_sorted_indices[bucket_index + 1]]) {
      ++bucket_index;
    }
    (*bucket_indices)[i] = bucket_index;
  }
}

void computeConditionalProbabilityTable(
    int bucket_count, const std::vector<int> &bucket_indices_x,
    const std::vector<int> &bucket_indices_y, std::vector<float> *table) {
  assert(bucket_indices_x.size() == bucket_indices_y.size());
  int sample_count = bucket_indices_x.size();
  table->clear();
  table->resize(bucket_count * bucket_count, 0);
  std::vector<int> bucket_sizes_x(bucket_count, 0);
  for (int i = 0; i < sample_count; ++i) {
    int x = bucket_indices_x[i];
    int y = bucket_indices_y[i];
    (*table)[x + bucket_count * y]++;
    bucket_sizes_x[x]++;
  }
  // Normalize each row to sum to one.
  for (int x = 0; x < bucket_count; ++x) {
    if (bucket_sizes_x[x] == 0) {
      continue;
    }
    for (int y = 0; y < bucket_count; ++y) {
      (*table)[x + bucket_count * y] /= bucket_sizes_x[x];
    }
  }
}

void printConditionalProbabilityTable(FILE *file, int bucket_count,
                                      const std::vector<float> &table) {
  const char *gamma_env = getenv("IREE_HOOK_GAMMA");
  float gamma = gamma_env ? strtof(gamma_env, nullptr) : 0.5f;

  const char *shades[] = {"  ", "_ ", "__", "▁_", "▁▁", "▂▁", "▂▂",
                          "▃▂", "▃▃", "▄▃", "▄▄", "▅▄", "▅▅", "▆▅",
                          "▆▆", "▇▆", "▇▇", "█▇", "██"};
  const int shades_count = (sizeof shades) / (sizeof shades[0]);
  fprintf(file, "        ");
  for (int x = 0; x < bucket_count; ++x) {
    fprintf(file, " %2x", x);
  }
  fprintf(file, "\n");
  for (int y = 0; y < bucket_count; ++y) {
    fprintf(file, "      %2x", y);
    for (int x = 0; x < bucket_count; ++x) {
      float probability = table[x + bucket_count * y];
      int shade_index = std::min(
          shades_count - 1,
          static_cast<int>(std::floor(std::pow(probability, gamma) *
                                      static_cast<float>(shades_count))));
      fprintf(file, " %s", shades[shade_index]);
    }
    fprintf(file, "\n");
  }
  fprintf(file, "\n");
}

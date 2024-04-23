// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef EXPERIMENTAL_HAL_EXECUTABLE_LIBRARY_CALL_HOOKS_STATS_H_
#define EXPERIMENTAL_HAL_EXECUTABLE_LIBRARY_CALL_HOOKS_STATS_H_

#include <cstdio>
#include <vector>

// Returns the sum of all elements.
float sum(const std::vector<float> &v);

// Returns the mean value.
float mean(const std::vector<float> &v);

// Returns the covariance between two vectors.
float covariance(const std::vector<float> &u, const std::vector<float> &v);

// Returns the variance of the vector.
float variance(const std::vector<float> &v);

// Returns the "Pearson correlation coefficient".
// Nothing particularly good about that choice of correlation metric.
// Just done naively.
float correlation(const std::vector<float> &u, const std::vector<float> &v);

// Split input data `v` into buckets i.e. (100/bucket_count)-percentiles.
void splitIntoBuckets(const std::vector<float> &v, int bucket_count,
                      std::vector<float> *bucket_means,
                      std::vector<int> *bucket_indices);

// Fills `table` with conditional probabilities of an index falling into a
// y-bucket given that it belongs to a x-bucket.
void computeConditionalProbabilityTable(
    int bucket_count, const std::vector<int> &bucket_indices_x,
    const std::vector<int> &bucket_indices_y, std::vector<float> *table);

// Render a conditional probability table as semi-graphical grayscale.
void printConditionalProbabilityTable(FILE *file, int bucket_count,
                                      const std::vector<float> &table);

#endif  // EXPERIMENTAL_HAL_EXECUTABLE_LIBRARY_CALL_HOOKS_STATS_H_

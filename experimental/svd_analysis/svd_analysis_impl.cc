#include "experimental/svd_analysis/svd_analysis_impl.h"

#include <Eigen/Core>
#include <Eigen/SVD>
#include <cmath>

extern "C" void svd_analysis_for_matrix_f32(FILE* file, const float* data,
                                            int rows, int cols) {
  using RowMajorMatrix =
      Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
  Eigen::VectorXf singularValues = RowMajorMatrix::Map(data, rows, cols)
                                       .bdcSvd()
                                       .singularValues()
                                       .cwiseMax(0.f);
  int size = singularValues.size();
  Eigen::VectorXf normalizedSingularValues;
  if (singularValues.maxCoeff() == 0.f) {
    normalizedSingularValues = Eigen::VectorXf::Zero(size);
  } else {
    normalizedSingularValues = singularValues / singularValues.maxCoeff();
  }
  std::vector<float> magnitude_boundaries = {1.f,   3e-1f, 1e-1f, 3e-2f, 1e-2f,
                                             3e-3f, 1e-3f, 3e-4f, 1e-4f, 3e-5f,
                                             1e-5f, 3e-6f, 1e-6f, 0.f};
  for (int b = 0; b < magnitude_boundaries.size() - 1; ++b) {
    int count = 0;
    for (int i = 0; i < size; ++i) {
      if (normalizedSingularValues[i] <= magnitude_boundaries[b] &&
          normalizedSingularValues[i] > magnitude_boundaries[b + 1]) {
        ++count;
      }
    }
    if (count) {
      fprintf(file,
              "       %6.1f%% of normalized singular values are <= %6.2g and > "
              "%6.2g\n",
              100.f * count / size, magnitude_boundaries[b],
              magnitude_boundaries[b + 1]);
    }
  }
  int count = 0;
  for (int i = 0; i < size; ++i) {
    if (normalizedSingularValues[i] == 0.f) {
      ++count;
    }
  }
  if (count) {
    fprintf(file, "       %6.1f%% of normalized singular values are == 0.\n",
            100.f * count / size);
  }
}

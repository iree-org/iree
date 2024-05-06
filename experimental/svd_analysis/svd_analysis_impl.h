#ifndef SVD_ANALYSIS_IMPL_H_
#define SVD_ANALYSIS_IMPL_H_

#include <stdio.h>

#ifdef __cplusplus
extern "C" {
#endif

void svd_analysis_for_matrix_f32(FILE* file, const float* data, int rows,
                                 int cols);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // SVD_ANALYSIS_IMPL_H_
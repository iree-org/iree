
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include "experimental/svd_analysis/svd_analysis_impl.h"
#include "iree/base/internal/math.h"
#include "iree/builtins/ukernel/mmt4d.h"
#include "iree/builtins/ukernel/mmt4d_internal.h"
#include "iree/hal/local/executable_library.h"

static float* allocate_and_convert_tiled_to_rowmajor_matrix_f32(
    iree_uk_type_t elem_type, const char* src_buffer, int offset, int stride,
    int rows1, int cols1, int rows0, int cols0) {
  int elem_size = iree_uk_type_size(elem_type);
  const char* src_data = src_buffer + elem_size * offset;
  int64_t dst_f32_data_size = 4 * rows1 * cols1 * rows0 * cols0;
  float* dst_f32_data = malloc(dst_f32_data_size);
  for (int64_t r1 = 0; r1 < rows1; ++r1) {
    for (int64_t c1 = 0; c1 < cols1; ++c1) {
      for (int64_t r0 = 0; r0 < rows0; ++r0) {
        for (int64_t c0 = 0; c0 < cols0; ++c0) {
          int64_t src_index = c0 + cols0 * (r0 + rows0 * c1) + stride * r1;
          float val = 0.f;
          switch (elem_type) {
            case IREE_UK_TYPE_FLOAT_32:
              val = ((const float*)src_data)[src_index];
              break;
            case IREE_UK_TYPE_FLOAT_16:
              val =
                  iree_math_f16_to_f32(((const uint16_t*)src_data)[src_index]);
              break;
            case IREE_UK_TYPE_BFLOAT_16:
              val =
                  iree_math_bf16_to_f32(((const uint16_t*)src_data)[src_index]);
              break;
            case IREE_UK_TYPE_SINT_32:
            case IREE_UK_TYPE_INT_32:
              val = ((const int32_t*)src_data)[src_index];
              break;
            case IREE_UK_TYPE_UINT_32:
              val = ((const uint32_t*)src_data)[src_index];
              break;
            case IREE_UK_TYPE_SINT_16:
            case IREE_UK_TYPE_INT_16:
              val = ((const int16_t*)src_data)[src_index];
              break;
            case IREE_UK_TYPE_UINT_16:
              val = ((const uint16_t*)src_data)[src_index];
              break;
            case IREE_UK_TYPE_SINT_8:
            case IREE_UK_TYPE_INT_8:
              val = ((const int8_t*)src_data)[src_index];
              break;
            case IREE_UK_TYPE_UINT_8:
              val = ((const uint8_t*)src_data)[src_index];
              break;
            case IREE_UK_TYPE_UINT_4:
              val = (((const uint8_t*)src_data)[src_index / 2] >>
                     ((src_index & 1) * 4)) &
                    0xf;
              break;
            default:
              assert(false && "unhandled element type");
              break;
          }
          int64_t dst_row = r1 * rows0 + r0;
          int64_t dst_col = c1 * cols0 + c0;
          int64_t dst_index = dst_row * cols1 * cols0 + dst_col;
          dst_f32_data[dst_index] = val;
        }
      }
    }
  }
  return dst_f32_data;
}

const char* iree_uk_type_category_str(iree_uk_type_t t) {
  switch (iree_uk_type_category(t)) {
    case IREE_UK_TYPE_CATEGORY_FLOAT_IEEE:
      return "f";
    case IREE_UK_TYPE_CATEGORY_FLOAT_BRAIN:
      return "bf";
    case IREE_UK_TYPE_CATEGORY_INTEGER_SIGNLESS:
      return "i";
    case IREE_UK_TYPE_CATEGORY_INTEGER_SIGNED:
      return "si";
    case IREE_UK_TYPE_CATEGORY_INTEGER_UNSIGNED:
      return "ui";
    default:
      assert(false && "unknown type");
      return "?";
  }
}

static void svd_analysis_for_tiled_matrix(FILE* file, const char* name,
                                          iree_uk_type_t elem_type,
                                          const char* src_buffer, int offset,
                                          int stride, int rows1, int cols1,
                                          int rows0, int cols0) {
  float* f32_data = allocate_and_convert_tiled_to_rowmajor_matrix_f32(
      elem_type, src_buffer, offset, stride, rows1, cols1, rows0, cols0);
  fprintf(file, "      %s: %dx%dx%dx%dx%s%d\n", name, rows1, cols1, rows0,
          cols0, iree_uk_type_category_str(elem_type),
          iree_uk_type_bit_count(elem_type));
  svd_analysis_for_matrix_f32(file, f32_data, rows1 * rows0, cols1 * cols0);
  free(f32_data);
}

#ifdef __GNUC__
#define IREE_HOOK_EXPORT __attribute__((visibility("default")))
#else
#define IREE_HOOK_EXPORT
#endif

const char* global_current_library_export_name = "";

IREE_HOOK_EXPORT void iree_uk_mmt4d_svd_analysis(
    const iree_uk_mmt4d_params_t* params) {
  iree_uk_mmt4d_type_t mmt4d_type = iree_uk_mmt4d_type(params->flags);
  iree_uk_type_t lhs_elem_type = iree_uk_mmt4d_lhs_type(mmt4d_type);
  iree_uk_type_t rhs_elem_type = iree_uk_mmt4d_rhs_type(mmt4d_type);
  printf("  CALL: %s\n", global_current_library_export_name);
  printf("    OP: mmt4d\n");
  svd_analysis_for_tiled_matrix(
      stdout, "LHS", lhs_elem_type, params->lhs_buffer, params->lhs_offset,
      params->lhs_stride0, params->M, params->K, params->M0, params->K0);
  svd_analysis_for_tiled_matrix(
      stdout, "RHS", rhs_elem_type, params->rhs_buffer, params->rhs_offset,
      params->rhs_stride0, params->N, params->K, params->N0, params->K0);
}

#ifdef IREE_HAL_EXECUTABLE_LIBRARY_CALL_HOOK
IREE_HOOK_EXPORT void iree_hal_executable_library_call_hook_begin(
    iree_string_view_t executable_identifier,
    const iree_hal_executable_library_v0_t* library, iree_host_size_t ordinal) {
  global_current_library_export_name = library->exports.names[ordinal];
}
IREE_HOOK_EXPORT void iree_hal_executable_library_call_hook_end(
    iree_string_view_t executable_identifier,
    const iree_hal_executable_library_v0_t* library, iree_host_size_t ordinal) {
  global_current_library_export_name = "";
}
#else
#error Need `cmake -DCMAKE_C_FLAGS=-DIREE_HAL_EXECUTABLE_LIBRARY_CALL_HOOK .`
#endif

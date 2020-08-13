// Copyright 2020 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef IREE_HAL_VMLA_OP_KERNELS_GENERIC_H_
#define IREE_HAL_VMLA_OP_KERNELS_GENERIC_H_

#include "absl/container/flat_hash_set.h"
#include "absl/container/inlined_vector.h"
#include "absl/types/span.h"
#include "iree/base/status.h"

namespace iree {
namespace hal {
namespace vmla {
namespace kernels {

template <typename T>
Status CompareEQ::Execute(absl::Span<const T> lhs_buffer,
                          absl::Span<const T> rhs_buffer,
                          absl::Span<uint8_t> dst_buffer) {
  for (size_t i = 0; i < dst_buffer.size(); ++i) {
    dst_buffer[i] = lhs_buffer[i] == rhs_buffer[i];
  }
  return OkStatus();
}

template <typename T>
Status CompareNE::Execute(absl::Span<const T> lhs_buffer,
                          absl::Span<const T> rhs_buffer,
                          absl::Span<uint8_t> dst_buffer) {
  for (size_t i = 0; i < dst_buffer.size(); ++i) {
    dst_buffer[i] = lhs_buffer[i] != rhs_buffer[i];
  }
  return OkStatus();
}

template <typename T>
Status CompareLT::Execute(absl::Span<const T> lhs_buffer,
                          absl::Span<const T> rhs_buffer,
                          absl::Span<uint8_t> dst_buffer) {
  for (size_t i = 0; i < dst_buffer.size(); ++i) {
    dst_buffer[i] = lhs_buffer[i] < rhs_buffer[i];
  }
  return OkStatus();
}

template <typename T>
Status CompareLE::Execute(absl::Span<const T> lhs_buffer,
                          absl::Span<const T> rhs_buffer,
                          absl::Span<uint8_t> dst_buffer) {
  for (size_t i = 0; i < dst_buffer.size(); ++i) {
    dst_buffer[i] = lhs_buffer[i] <= rhs_buffer[i];
  }
  return OkStatus();
}

template <typename T>
Status CompareGT::Execute(absl::Span<const T> lhs_buffer,
                          absl::Span<const T> rhs_buffer,
                          absl::Span<uint8_t> dst_buffer) {
  for (size_t i = 0; i < dst_buffer.size(); ++i) {
    dst_buffer[i] = lhs_buffer[i] > rhs_buffer[i];
  }
  return OkStatus();
}

template <typename T>
Status CompareGE::Execute(absl::Span<const T> lhs_buffer,
                          absl::Span<const T> rhs_buffer,
                          absl::Span<uint8_t> dst_buffer) {
  for (size_t i = 0; i < dst_buffer.size(); ++i) {
    dst_buffer[i] = lhs_buffer[i] >= rhs_buffer[i];
  }
  return OkStatus();
}

namespace impl {
inline absl::InlinedVector<size_t, 6> ComputeCopyStrides(ShapeSpan shape,
                                                         size_t element_size) {
  absl::InlinedVector<size_t, 6> strides(shape.size());
  strides.back() = element_size;
  for (int i = shape.size() - 2; i >= 0; --i) {
    strides[i] = strides[i + 1] * shape[i + 1];
  }
  return strides;
}

inline void CopyRegion(absl::Span<const uint8_t> src_buffer,
                       absl::Span<const size_t> src_strides,
                       absl::Span<const int32_t> src_indices,
                       absl::Span<uint8_t> dst_buffer,
                       absl::Span<const size_t> dst_strides,
                       absl::Span<const int32_t> dst_indices,
                       absl::Span<const int32_t> lengths) {
  if (lengths.size() > 1) {
    for (int i = 0; i < lengths[0]; ++i) {
      size_t src_offset = src_strides[0] * (src_indices[0] + i);
      size_t dst_offset = dst_strides[0] * (dst_indices[0] + i);
      CopyRegion(src_buffer.subspan(src_offset), src_strides.subspan(1),
                 src_indices.subspan(1), dst_buffer.subspan(dst_offset),
                 dst_strides.subspan(1), dst_indices.subspan(1),
                 lengths.subspan(1));
    }
  } else {
    DCHECK_EQ(dst_strides.size(), 1);
    DCHECK_EQ(src_strides.size(), 1);
    DCHECK_EQ(src_indices.size(), 1);
    DCHECK_EQ(dst_indices.size(), 1);
    DCHECK_EQ(lengths.size(), 1);
    auto src_offset = src_indices[0] * src_strides[0];
    auto dst_offset = dst_indices[0] * dst_strides[0];
    auto length = dst_strides[0] * lengths[0];
    std::memcpy(dst_buffer.data() + dst_offset, src_buffer.data() + src_offset,
                length);
  }
}
}  // namespace impl

// TODO(benvanik): replace with a real implementation once copy is defined.
// TODO(gcmn): More consistent/principled handling for scalars.
template <int element_size>
Status Copy::Execute(absl::Span<const uint8_t> src_buffer, ShapeSpan src_shape,
                     absl::Span<const int32_t> src_indices,
                     absl::Span<uint8_t> dst_buffer, ShapeSpan dst_shape,
                     absl::Span<const int32_t> dst_indices,
                     absl::Span<const int32_t> lengths) {
  DCHECK_EQ(src_indices.size(), lengths.size());
  DCHECK_EQ(dst_indices.size(), lengths.size());
  DCHECK_EQ(src_shape.size(), lengths.size());
  DCHECK_EQ(dst_shape.size(), lengths.size());
  if (lengths.empty()) {
    std::memcpy(dst_buffer.data(), src_buffer.data(), element_size);
    return OkStatus();
  }

  // TODO(gcmn) Maybe we can fast-path earlier if we detect contiguous memory
  // across multiple rows.
  auto src_strides = impl::ComputeCopyStrides(src_shape, element_size);
  auto dst_strides = impl::ComputeCopyStrides(dst_shape, element_size);
  DCHECK_EQ(src_strides.size(), lengths.size());
  DCHECK_EQ(dst_strides.size(), lengths.size());
  impl::CopyRegion(src_buffer, src_strides, src_indices, dst_buffer,
                   dst_strides, dst_indices, lengths);
  return OkStatus();
}

template <typename T>
Status Conv2D::Execute(absl::Span<const T> input_buffer, ShapeSpan input_shape,
                       absl::Span<const T> filter_buffer,
                       ShapeSpan filter_shape, absl::Span<T> dst_buffer,
                       ShapeSpan dst_shape, ShapeSpan window_strides,
                       ShapeSpan pad_h, ShapeSpan pad_w, ShapeSpan dilation,
                       const int32_t groups) {
  const std::array<int32_t, 3> input_strides = {input_shape[1] * input_shape[2],
                                                input_shape[2], 1};
  const std::array<int32_t, 4> filter_strides = {
      filter_shape[1] * filter_shape[2] * filter_shape[3],
      filter_shape[2] * filter_shape[3], filter_shape[3], 1};
  const std::array<int32_t, 3> dst_strides = {dst_shape[1] * dst_shape[2],
                                              dst_shape[2], 1};
  // Direct 2d (grouped) convolution slow implementation. ref:
  // https://www.tensorflow.org/versions/r2.0/api_docs/python/tf/nn/convolution)
  // TODO(ataei): Implement tiled GEMM based implementation.
  const int output_group_size = dst_shape[2] / groups;
  const int input_group_size = input_shape[2] / groups;
  for (int ho = 0; ho < dst_shape[0]; ho++) {
    for (int wo = 0; wo < dst_shape[1]; wo++) {
      for (int g = 0; g < groups; ++g) {
        for (int co = 0; co < output_group_size; co++) {
          const int cg_o = g * output_group_size + co;
          const int y_i = ho * dst_strides[0] + wo * dst_strides[1] + cg_o;
          T dst_value = T(0);
          for (int ci = 0; ci < input_group_size; ci++) {
            for (int kh = 0; kh < filter_shape[0]; kh++) {
              const int ih = ho * window_strides[0] + kh - pad_h[0];
              // left-right padding condition.
              if (ih < 0 || ih >= input_shape[0]) continue;
              for (int kw = 0; kw < filter_shape[1]; kw++) {
                // top-bottom padding condition.
                const int iw = wo * window_strides[1] + kw - pad_w[0];
                if (iw < 0 || iw >= input_shape[1]) continue;
                const int cg_i = g * input_group_size + ci;
                const int w_i = kh * dilation[0] * filter_strides[0] +
                                kw * dilation[1] * filter_strides[1] +
                                cg_i * filter_strides[2] + co;
                const int x_i =
                    ih * input_strides[0] + iw * input_strides[1] + cg_i;
                dst_value += input_buffer[x_i] * filter_buffer[w_i];
              }
            }
          }
          dst_buffer[y_i] = dst_value;
        }
      }
    }
  }
  return OkStatus();
}

template <typename T>
Status Select::Execute(absl::Span<const uint8_t> cond_buffer,
                       absl::Span<const T> lhs_buffer,
                       absl::Span<const T> rhs_buffer,
                       absl::Span<T> dst_buffer) {
  for (size_t i = 0; i < dst_buffer.size(); ++i) {
    dst_buffer[i] = cond_buffer[i] ? lhs_buffer[i] : rhs_buffer[i];
  }
  return OkStatus();
}

template <typename T>
Status Transpose::Execute(absl::Span<const T> src_buffer,
                          absl::Span<T> dst_buffer, ShapeSpan src_shape,
                          absl::Span<const int32_t> perm) {
  // This implementation is .... not fast.
  int rank = src_shape.size();
  absl::InlinedVector<int, 8> src_strides(rank);
  absl::InlinedVector<int, 8> dst_strides(rank);
  size_t src_stride = 1;
  size_t dst_stride = 1;
  for (int dim_i = rank - 1; dim_i >= 0; --dim_i) {
    src_strides[dim_i] = src_stride;
    dst_strides[dim_i] = dst_stride;
    src_stride *= src_shape[dim_i];
    dst_stride *= src_shape[perm[dim_i]];
  }
  for (size_t dst_i = 0; dst_i < dst_buffer.size(); ++dst_i) {
    size_t src_i = 0;
    size_t t = dst_i;
    for (int dim_i = 0; dim_i < rank; ++dim_i) {
      size_t ratio = t / dst_strides[dim_i];
      t -= ratio * dst_strides[dim_i];
      src_i += ratio * src_strides[perm[dim_i]];
    }
    dst_buffer[dst_i] = src_buffer[src_i];
  }
  return OkStatus();
}

namespace impl {
inline void IncrementShapeIndex(absl::Span<int32_t> indices, ShapeSpan shape) {
  for (int i = indices.size() - 1; i >= 0; --i) {
    if (++indices[i] < shape[i]) return;
    indices[i] = 0;
  }
}

inline bool IsPadding(absl::Span<const int32_t> indices, ShapeSpan shape,
                      absl::Span<const int32_t> edge_padding_low,
                      absl::Span<const int32_t> edge_padding_high,
                      absl::Span<const int32_t> interior_padding) {
  for (int i = 0; i < indices.size(); ++i) {
    auto index = indices[i];
    if (index < edge_padding_low[i] ||
        index >= shape[i] - edge_padding_high[i] ||
        (index - edge_padding_low[i]) % (interior_padding[i] + 1) != 0) {
      return true;
    }
  }

  return false;
}
}  // namespace impl

template <typename T>
Status Pad::Execute(absl::Span<const T> src_buffer,
                    absl::Span<const T> padding_value_buffer,
                    absl::Span<T> dst_buffer, ShapeSpan src_shape,
                    ShapeSpan dst_shape,
                    absl::Span<const int32_t> edge_padding_low,
                    absl::Span<const int32_t> edge_padding_high,
                    absl::Span<const int32_t> interior_padding) {
  // This implementation is not at all fast, as it iterates every index in the
  // destination buffer individually. Potential improvements:
  // 1. Fill the dst buffer with padded value initially. Only need to iterate
  //    through source buffer and can exit early.
  // 2. Use striding to advance through larger swaths of the buffer with a
  //    memcpy from src and filling (or skipping) padded incides. Especially
  //    useful when e.g. entire rows are padded.

  // TODO(b/140836672) support negative padding

  if (padding_value_buffer.size() != 1) {
    return InvalidArgumentErrorBuilder(IREE_LOC)
           << "Padding value buffer is larger than one element.";
  }
  auto padding_value = padding_value_buffer.front();

  absl::InlinedVector<int, 8> dst_indices(src_shape.size(), 0);

  const T* src_ptr = src_buffer.begin();
  T* dst_ptr = dst_buffer.begin();
  while (dst_ptr != dst_buffer.end()) {
    if (impl::IsPadding(dst_indices, dst_shape, edge_padding_low,
                        edge_padding_high, interior_padding)) {
      *dst_ptr++ = padding_value;
    } else {
      DCHECK(src_ptr != src_buffer.end());
      *dst_ptr++ = *src_ptr++;
    }
    impl::IncrementShapeIndex(absl::MakeSpan(dst_indices), dst_shape);
  }

  return OkStatus();
}

template <typename T>
Status Gather::Execute(absl::Span<const T> src_buffer,
                       absl::Span<const int32_t> indices_buffer,
                       absl::Span<T> dst_buffer, ShapeSpan src_shape,
                       ShapeSpan indices_shape, ShapeSpan dst_shape,
                       const int32_t dim, const int32_t batch_dims) {
  std::vector<int32_t> output_strides(dst_shape.size(), 1);
  std::vector<int32_t> input_strides(src_shape.size(), 1);
  std::vector<int32_t> indices_strides(indices_shape.size(), 1);
  auto compute_strides = [](ShapeSpan shape, std::vector<int32_t>& strides) {
    for (int i = shape.size() - 2; i >= 0; --i) {
      strides[i] = strides[i + 1] * shape[i + 1];
    }
  };
  compute_strides(dst_shape, output_strides);
  compute_strides(src_shape, input_strides);
  compute_strides(indices_shape, indices_strides);
  size_t outer_size = 1;
  for (size_t i = 0; i < dim; ++i) {
    outer_size *= src_shape[i];
  }
  // stride for batch outer dims.
  size_t batch_stride = 1;
  for (size_t i = batch_dims; i > 0; --i) {
    batch_stride *= src_shape[i];
  }
  const size_t input_stride =
      dim > 0 ? input_strides[dim - 1] : input_strides[0];
  const size_t output_stride =
      dim > 0 ? output_strides[dim - 1] : output_strides[0];
  const size_t slize_size = input_strides[dim];
  const int indices_size =
      indices_shape.size() == 0
          ? 1
          : indices_shape[batch_dims] * indices_strides[batch_dims];
  // This is equivalent to the linearized version of followng array expression:
  // clang-format off
  // dst[d_0,...,d_{dim-1},                     i_B,...,i_{M-1}, d_{dim+1},...,d_{N-1}] =
  // src[d_0,...,d_{dim-1},indices[d_0,...,d_1, i_B,...,i_{M-1}, d_{dim+1},...,d_{N-1}]
  // clang-format on
  // see:https://www.tensorflow.org/api_docs/python/tf/gather
  // TODO(ataei): Shrink inner loop by scanning indices_buffer for
  // contiguous indices and collide the copy of these slices.
  for (size_t i = 0; i < outer_size; ++i) {
    const int batch_offset =
        batch_dims == 0 ? 0 : (i / batch_stride) * indices_strides[batch_dims];
    for (size_t j = 0; j < indices_size; ++j) {
      const size_t dst_offset = i * output_stride + j * slize_size;
      const size_t src_offset =
          i * input_stride + indices_buffer[batch_offset + j] * slize_size;
      std::memcpy(dst_buffer.data() + dst_offset,
                  src_buffer.data() + src_offset, sizeof(T) * slize_size);
    }
  }
  return OkStatus();
}

namespace impl {
template <typename T>
Status ScatterCopy(absl::Span<const T> src_buffer, absl::Span<T> dst_buffer,
                   ShapeSpan src_shape, ShapeSpan dst_shape) {
  if (src_shape.empty()) {
    dst_buffer[0] = src_buffer[0];
    return OkStatus();
  }

  // Scatter cannot subscatter, it must be legal across he entire shape.
  // Therefore if the src and dst shape match we can copy the full bytes over.
  if (src_shape == dst_shape) {
    memcpy(dst_buffer.data(), src_buffer.data(), src_buffer.size() * sizeof(T));
    return OkStatus();
  }

  auto src_stride = 1;
  for (auto size : src_shape.subspan(1)) {
    src_stride *= size;
  }

  auto dst_stride = 1;
  for (auto size : dst_shape.subspan(1)) {
    dst_stride *= size;
  }

  for (int i = 0; i < src_shape[0]; i++) {
    IREE_RETURN_IF_ERROR(
        ScatterCopy(src_buffer.subspan(i * src_stride, src_stride),
                    dst_buffer.subspan(i * dst_stride, dst_stride),
                    src_shape.subspan(1), dst_shape.subspan(1)));
  }

  return OkStatus();
}

// Scatter helper compute the offset into src buffer, removing the dependency
// on the indices buffer.
template <typename T>
Status ScatterHelper(absl::Span<const T> src_buffer,
                     absl::Span<const int32_t> indices_buffer,
                     absl::Span<T> dst_buffer, ShapeSpan src_shape,
                     ShapeSpan dst_shape) {
  size_t offset = 0;
  for (int i = 0; i < indices_buffer.size(); i++) {
    offset = offset * dst_shape[i] + indices_buffer[i];
  }

  for (int i = indices_buffer.size(); i < dst_shape.size(); i++) {
    offset *= dst_shape[i];
  }

  if ((src_shape.size() + indices_buffer.size()) != dst_shape.size()) {
    return InvalidArgumentErrorBuilder(IREE_LOC)
           << "Attempting to scatter to differing dimensions.";
  }

  IREE_RETURN_IF_ERROR(ScatterCopy(src_buffer, dst_buffer.subspan(offset),
                                   src_shape,
                                   dst_shape.subspan(indices_buffer.size())));

  return OkStatus();
}
}  // namespace impl

template <typename T>
Status Scatter::Execute(absl::Span<const T> src_buffer,
                        absl::Span<const int32_t> indices_buffer,
                        absl::Span<T> dst_buffer, ShapeSpan src_shape,
                        ShapeSpan indices_shape, ShapeSpan dst_shape) {
  int indices_rank = indices_shape.size();

  // First dimension of indices is the batch update.
  int32_t batch_size = 1;
  if (indices_rank > 0) {
    batch_size = indices_shape[0];
  }

  // Secnd dimensions of indices is the indice offset to scatter along.
  int32_t indices_size = 1;
  if (indices_rank > 1) {
    indices_size = indices_shape[1];
  }

  // Compute the source size per scatter.
  int32_t src_size = 1;
  for (auto val : src_shape.subspan(1)) {
    src_size *= val;
  }

  for (int i = 0; i < batch_size; i++) {
    IREE_RETURN_IF_ERROR(impl::ScatterHelper(
        src_buffer.subspan(i * src_size, src_size),
        indices_buffer.subspan(i * indices_size, indices_size), dst_buffer,
        src_shape.subspan(1), dst_shape));
  }

  return OkStatus();
}

template <typename T>
Status Reverse::Execute(absl::Span<const T> src_buffer,
                        absl::Span<T> dst_buffer, ShapeSpan src_shape,
                        absl::Span<const int32_t> dimensions) {
  // This implementation is not fast either.
  int rank = src_shape.size();
  absl::InlinedVector<int, 8> strides(rank);
  size_t stride = 1;
  for (int dim_i = rank - 1; dim_i >= 0; --dim_i) {
    strides[dim_i] = stride;
    stride *= src_shape[dim_i];
  }
  absl::flat_hash_set<int32_t> dims_set(dimensions.begin(), dimensions.end());
  for (size_t dst_i = 0; dst_i < dst_buffer.size(); ++dst_i) {
    size_t src_i = 0;
    size_t t = dst_i;
    for (int dim_i = 0; dim_i < rank; ++dim_i) {
      size_t ratio = t / strides[dim_i];
      t -= ratio * strides[dim_i];
      bool do_reverse = dims_set.contains(dim_i);
      src_i += (do_reverse ? (src_shape[dim_i] - 1 - ratio) : ratio) *
               strides[dim_i];
    }
    dst_buffer[dst_i] = src_buffer[src_i];
  }
  return OkStatus();
}

template <typename T>
Status Broadcast::Execute(absl::Span<const T> src_buffer,
                          absl::Span<T> dst_buffer) {
  for (size_t i = 0; i < dst_buffer.size(); ++i) {
    dst_buffer[i] = src_buffer[0];
  }
  return OkStatus();
}

template <typename T>
Status Tile::Execute(absl::Span<const T> src_buffer, absl::Span<T> dst_buffer,
                     ShapeSpan src_shape, ShapeSpan dst_shape) {
  // This implementation is .... not fast.
  int rank = dst_shape.size();
  absl::InlinedVector<int, 8> src_strides(rank);
  absl::InlinedVector<int, 8> dst_strides(rank);
  size_t src_stride = 1;
  size_t dst_stride = 1;
  for (int dim_i = rank - 1; dim_i >= 0; --dim_i) {
    src_strides[dim_i] = src_stride;
    dst_strides[dim_i] = dst_stride;
    src_stride *= src_shape[dim_i];
    dst_stride *= dst_shape[dim_i];
  }
  for (size_t dst_i = 0; dst_i < dst_buffer.size(); ++dst_i) {
    size_t src_i = 0;
    size_t t = dst_i;
    for (int dim_i = 0; dim_i < rank; ++dim_i) {
      src_i += t / dst_strides[dim_i] % src_shape[dim_i] * src_strides[dim_i];
      t %= dst_strides[dim_i];
    }
    dst_buffer[dst_i] = src_buffer[src_i];
  }
  return OkStatus();
}

template <typename T>
Status Not::Execute(absl::Span<const T> src_buffer, absl::Span<T> dst_buffer) {
  for (size_t i = 0; i < dst_buffer.size(); ++i) {
    dst_buffer[i] = ~src_buffer[i];
  }
  return OkStatus();
}

template <typename T>
Status And::Execute(absl::Span<const T> lhs_buffer,
                    absl::Span<const T> rhs_buffer, absl::Span<T> dst_buffer) {
  for (size_t i = 0; i < dst_buffer.size(); ++i) {
    dst_buffer[i] = lhs_buffer[i] & rhs_buffer[i];
  }
  return OkStatus();
}

template <typename T>
Status Or::Execute(absl::Span<const T> lhs_buffer,
                   absl::Span<const T> rhs_buffer, absl::Span<T> dst_buffer) {
  for (size_t i = 0; i < dst_buffer.size(); ++i) {
    dst_buffer[i] = lhs_buffer[i] | rhs_buffer[i];
  }
  return OkStatus();
}

template <typename T>
Status Xor::Execute(absl::Span<const T> lhs_buffer,
                    absl::Span<const T> rhs_buffer, absl::Span<T> dst_buffer) {
  for (size_t i = 0; i < dst_buffer.size(); ++i) {
    dst_buffer[i] = lhs_buffer[i] ^ rhs_buffer[i];
  }
  return OkStatus();
}

template <typename T>
Status ShiftLeft::Execute(absl::Span<const T> lhs_buffer,
                          absl::Span<const T> rhs_buffer,
                          absl::Span<T> dst_buffer) {
  for (size_t i = 0; i < dst_buffer.size(); ++i) {
    dst_buffer[i] = lhs_buffer[i] << rhs_buffer[i];
  }
  return OkStatus();
}

template <typename T>
Status ShiftRight::Execute(absl::Span<const T> lhs_buffer,
                           absl::Span<const T> rhs_buffer,
                           absl::Span<T> dst_buffer) {
  for (size_t i = 0; i < dst_buffer.size(); ++i) {
    dst_buffer[i] = lhs_buffer[i] >> rhs_buffer[i];
  }
  return OkStatus();
}

template <typename T>
Status Add::Execute(absl::Span<const T> lhs_buffer,
                    absl::Span<const T> rhs_buffer, absl::Span<T> dst_buffer) {
  for (size_t i = 0; i < dst_buffer.size(); ++i) {
    dst_buffer[i] = lhs_buffer[i] + rhs_buffer[i];
  }
  return OkStatus();
}

template <typename T>
Status Sub::Execute(absl::Span<const T> lhs_buffer,
                    absl::Span<const T> rhs_buffer, absl::Span<T> dst_buffer) {
  for (size_t i = 0; i < dst_buffer.size(); ++i) {
    dst_buffer[i] = lhs_buffer[i] - rhs_buffer[i];
  }
  return OkStatus();
}

template <typename T>
Status Abs::Execute(absl::Span<const T> src_buffer, absl::Span<T> dst_buffer) {
  for (size_t i = 0; i < dst_buffer.size(); ++i) {
    dst_buffer[i] = std::abs(src_buffer[i]);
  }
  return OkStatus();
}

template <typename T>
Status Neg::Execute(absl::Span<const T> src_buffer, absl::Span<T> dst_buffer) {
  for (size_t i = 0; i < dst_buffer.size(); ++i) {
    dst_buffer[i] = -src_buffer[i];
  }
  return OkStatus();
}

template <typename T>
Status Mul::Execute(absl::Span<const T> lhs_buffer,
                    absl::Span<const T> rhs_buffer, absl::Span<T> dst_buffer) {
  for (size_t i = 0; i < dst_buffer.size(); ++i) {
    dst_buffer[i] = lhs_buffer[i] * rhs_buffer[i];
  }
  return OkStatus();
}

template <typename T>
Status Div::Execute(absl::Span<const T> lhs_buffer,
                    absl::Span<const T> rhs_buffer, absl::Span<T> dst_buffer) {
  for (size_t i = 0; i < dst_buffer.size(); ++i) {
    dst_buffer[i] = lhs_buffer[i] / rhs_buffer[i];
  }
  return OkStatus();
}

template <typename T>
Status Rem::Execute(absl::Span<const T> lhs_buffer,
                    absl::Span<const T> rhs_buffer, absl::Span<T> dst_buffer) {
  for (size_t i = 0; i < dst_buffer.size(); ++i) {
    dst_buffer[i] = remainder(lhs_buffer[i], rhs_buffer[i]);
  }
  return OkStatus();
}

template <typename T>
Status Pow::Execute(absl::Span<const T> lhs_buffer,
                    absl::Span<const T> rhs_buffer, absl::Span<T> dst_buffer) {
  for (size_t i = 0; i < dst_buffer.size(); ++i) {
    dst_buffer[i] = std::pow(lhs_buffer[i], rhs_buffer[i]);
  }
  return OkStatus();
}

template <typename T>
Status Exp::Execute(absl::Span<const T> src_buffer, absl::Span<T> dst_buffer) {
  for (size_t i = 0; i < dst_buffer.size(); ++i) {
    dst_buffer[i] = std::exp(src_buffer[i]);
  }
  return OkStatus();
}

template <typename T>
Status Rsqrt::Execute(absl::Span<const T> src_buffer,
                      absl::Span<T> dst_buffer) {
  for (size_t i = 0; i < dst_buffer.size(); ++i) {
    dst_buffer[i] = 1.0 / std::sqrt(src_buffer[i]);
  }
  return OkStatus();
}

template <typename T>
Status Sqrt::Execute(absl::Span<const T> src_buffer, absl::Span<T> dst_buffer) {
  for (size_t i = 0; i < dst_buffer.size(); ++i) {
    dst_buffer[i] = std::sqrt(src_buffer[i]);
  }
  return OkStatus();
}

template <typename T>
Status Log::Execute(absl::Span<const T> src_buffer, absl::Span<T> dst_buffer) {
  for (size_t i = 0; i < dst_buffer.size(); ++i) {
    dst_buffer[i] = std::log(src_buffer[i]);
  }
  return OkStatus();
}

template <typename T>
Status Cos::Execute(absl::Span<const T> src_buffer, absl::Span<T> dst_buffer) {
  for (size_t i = 0; i < dst_buffer.size(); ++i) {
    dst_buffer[i] = std::cos(src_buffer[i]);
  }
  return OkStatus();
}

template <typename T>
Status Sin::Execute(absl::Span<const T> src_buffer, absl::Span<T> dst_buffer) {
  for (size_t i = 0; i < dst_buffer.size(); ++i) {
    dst_buffer[i] = std::sin(src_buffer[i]);
  }
  return OkStatus();
}

template <typename T>
Status Tanh::Execute(absl::Span<const T> src_buffer, absl::Span<T> dst_buffer) {
  for (size_t i = 0; i < dst_buffer.size(); ++i) {
    dst_buffer[i] = std::tanh(src_buffer[i]);
  }
  return OkStatus();
}

template <typename T>
Status Atan2::Execute(absl::Span<const T> lhs_buffer,
                      absl::Span<const T> rhs_buffer,
                      absl::Span<T> dst_buffer) {
  for (size_t i = 0; i < dst_buffer.size(); ++i) {
    dst_buffer[i] = std::atan2(lhs_buffer[i], rhs_buffer[i]);
  }
  return OkStatus();
}

template <typename T>
Status Min::Execute(absl::Span<const T> lhs_buffer,
                    absl::Span<const T> rhs_buffer, absl::Span<T> dst_buffer) {
  for (size_t i = 0; i < dst_buffer.size(); ++i) {
    dst_buffer[i] = std::min(lhs_buffer[i], rhs_buffer[i]);
  }
  return OkStatus();
}

template <typename T>
Status Max::Execute(absl::Span<const T> lhs_buffer,
                    absl::Span<const T> rhs_buffer, absl::Span<T> dst_buffer) {
  for (size_t i = 0; i < dst_buffer.size(); ++i) {
    dst_buffer[i] = std::max(lhs_buffer[i], rhs_buffer[i]);
  }
  return OkStatus();
}

template <typename T>
Status Clamp::Execute(absl::Span<const T> min_buffer,
                      absl::Span<const T> src_buffer,
                      absl::Span<const T> max_buffer,
                      absl::Span<T> dst_buffer) {
  for (size_t i = 0; i < dst_buffer.size(); ++i) {
    T src = src_buffer[i];
    T min = min_buffer[i];
    T max = max_buffer[i];
    dst_buffer[i] = src <= min ? min : src >= max ? max : src;
  }
  return OkStatus();
}

template <typename T>
Status Floor::Execute(absl::Span<const T> src_buffer,
                      absl::Span<T> dst_buffer) {
  for (size_t i = 0; i < dst_buffer.size(); ++i) {
    dst_buffer[i] = std::floor(src_buffer[i]);
  }
  return OkStatus();
}

template <typename T>
Status Ceil::Execute(absl::Span<const T> src_buffer, absl::Span<T> dst_buffer) {
  for (size_t i = 0; i < dst_buffer.size(); ++i) {
    dst_buffer[i] = std::ceil(src_buffer[i]);
  }
  return OkStatus();
}

template <typename SRC, typename DST>
Status Convert::Execute(absl::Span<const SRC> src_buffer,
                        absl::Span<DST> dst_buffer) {
  DCHECK_EQ(src_buffer.size(), dst_buffer.size());
  for (size_t i = 0; i < dst_buffer.size(); ++i) {
    dst_buffer[i] = static_cast<DST>(src_buffer[i]);
  }
  return OkStatus();
}

namespace impl {

struct SumKernel {
  template <typename T>
  inline void operator()(T* value0, const T value1) {
    *value0 += value1;
  }
};

struct MinKernel {
  template <typename T>
  inline void operator()(T* value0, const T value1) {
    *value0 = std::min(*value0, value1);
  }
};

struct MaxKernel {
  template <typename T>
  inline void operator()(T* value0, const T value1) {
    *value0 = std::max(*value0, value1);
  }
};

template <typename T, typename KernelImpl>
inline void ReduceDimension(absl::Span<const T> src_buffer,
                            absl::Span<T> dst_buffer, ShapeSpan src_shape,
                            absl::Span<const int32_t> reduce_dims,
                            absl::Span<const int> dst_strides, int dim,
                            absl::Span<int> src_indices, size_t flat_src_i,
                            size_t src_stride) {
  if (dim < 0) {
    // Base case of the recursion - figure out which elements should be acted
    // upon and apply the reduction kernel to them.

    // Derive destination indices from source indices.
    // For example,
    //     reduce_dims: [1, 2]
    //     src_indices: [2, 1, 3, 0]
    //                      ^  ^
    //                      |  |
    //                      |----- remove these dimensions
    //     dst_indices: [2, 0]
    //
    // TODO(scotttodd): Clean this up somehow, share across recursion levels?
    size_t dst_size = src_shape.size() - reduce_dims.size();
    absl::InlinedVector<int, 8> dst_indices;
    for (size_t i = 0; i < src_indices.size(); ++i) {
      if (std::find(std::begin(reduce_dims), std::end(reduce_dims), i) ==
          std::end(reduce_dims)) {
        dst_indices.push_back(src_indices[i]);
      }
    }
    // Compute the flattened index into dst_buffer at [dst_indices].
    size_t dst_i = 0;
    for (size_t i = 0; i < dst_indices.size(); ++i) {
      dst_i += dst_indices[i] * dst_strides[dst_size - 1 - i];
    }

    // Flattened src and dst indices have been computed, invoke the kernel.
    KernelImpl()(&dst_buffer[dst_i], src_buffer[flat_src_i]);
    return;
  }

  // Iterate through the current dimension in the source shape, recursing
  // down one dimension at a time.
  //
  // This touches each element in the source buffer once, tracking complete
  // dimensions within the shaped source buffer and using them to compute
  // the corresponding indices (shaped and flattened) within the destination
  // buffer. Each element in the destination buffer will be touched multiple
  // times.
  //
  // Note that cache coherency isn't considered here, and some computations
  // are redundant, so this could be optimized substantially.
  for (size_t dim_i = 0; dim_i < src_shape[dim]; ++dim_i) {
    src_indices[dim] = dim_i;

    // Recurse down to the next dimension (e.g. 2 -> 1 -> 0 -> base case)
    //   * Add the current stride to flat_src_i
    //   * Multiply src_stride by this dimension's shape
    ReduceDimension<T, KernelImpl>(src_buffer, dst_buffer, src_shape,
                                   reduce_dims, dst_strides, dim - 1,
                                   src_indices, flat_src_i + dim_i * src_stride,
                                   src_stride * src_shape[dim]);
  }
}

template <typename T, typename KernelImpl>
Status GenericReduce(absl::Span<const T> src_buffer,
                     absl::Span<const T> init_buffer, absl::Span<T> dst_buffer,
                     int32_t dimension, ShapeSpan src_shape,
                     ShapeSpan dst_shape) {
  // Initialize using init_buffer, which is expected to be a scalar.
  std::fill_n(dst_buffer.data(), dst_buffer.size(), init_buffer[0]);

  // Precompute destination strides.
  int dst_rank = dst_shape.size();
  absl::InlinedVector<int, 8> dst_strides;
  size_t dst_stride = 1;
  for (int dim_i = dst_rank - 1; dim_i >= 0; --dim_i) {
    dst_strides.push_back(dst_stride);
    dst_stride *= dst_shape[dim_i];
  }

  // Call the helper (recursive) function, starting with:
  //   * source index [0, 0, ..., 0]
  //   * the innermost dimension (last in the shape)
  //   * flat_src_i of 0 (corresponds to [0, 0, ..., 0] above)
  //   * source stride 1
  absl::InlinedVector<int, 8> src_indices(src_shape.size(), 0);
  ReduceDimension<T, KernelImpl>(src_buffer, dst_buffer, src_shape, {dimension},
                                 absl::MakeSpan(dst_strides),
                                 src_shape.size() - 1,
                                 absl::MakeSpan(src_indices), 0, 1);

  return OkStatus();
}

}  // namespace impl

template <typename T>
Status ReduceSum::Execute(absl::Span<const T> src_buffer,
                          absl::Span<const T> init_buffer,
                          absl::Span<T> dst_buffer, int32_t dimension,
                          ShapeSpan src_shape, ShapeSpan dst_shape) {
  return impl::GenericReduce<T, impl::SumKernel>(
      src_buffer, init_buffer, dst_buffer, dimension, src_shape, dst_shape);
}

template <typename T>
Status ReduceMin::Execute(absl::Span<const T> src_buffer,
                          absl::Span<const T> init_buffer,
                          absl::Span<T> dst_buffer, int32_t dimension,
                          ShapeSpan src_shape, ShapeSpan dst_shape) {
  return impl::GenericReduce<T, impl::MinKernel>(
      src_buffer, init_buffer, dst_buffer, dimension, src_shape, dst_shape);
}

template <typename T>
Status ReduceMax::Execute(absl::Span<const T> src_buffer,
                          absl::Span<const T> init_buffer,
                          absl::Span<T> dst_buffer, int32_t dimension,
                          ShapeSpan src_shape, ShapeSpan dst_shape) {
  return impl::GenericReduce<T, impl::MaxKernel>(
      src_buffer, init_buffer, dst_buffer, dimension, src_shape, dst_shape);
}

namespace impl {

template <typename T, typename KernelImpl>
void ComputePoolingWindow(absl::Span<const T> src_buffer,
                          absl::Span<const int> src_indices,
                          ShapeSpan src_shape, T init_value,
                          ShapeSpan window_dimensions, T* dst_value) {
  int rank = src_shape.size();
  absl::InlinedVector<int, 8> window_indices(rank, 0);
  auto getSrcValue = [&]() -> T {
    int flat_idx = 0;
    for (int i = 0; i < rank; ++i) {
      int idx = src_indices[i] + window_indices[i];
      if (idx < 0 || idx >= src_shape[i]) return init_value;
      flat_idx = flat_idx * src_shape[i] + idx;
    }
    return src_buffer[flat_idx];
  };

  *dst_value = init_value;
  for (int i = 0, e = GetElementCount(window_dimensions); i < e; ++i) {
    KernelImpl()(dst_value, getSrcValue());
    IncrementShapeIndex(absl::MakeSpan(window_indices), window_dimensions);
  }
}

template <typename T, typename KernelImpl>
Status GenericPooling(absl::Span<const T> src_buffer,
                      absl::Span<const T> init_buffer, absl::Span<T> dst_buffer,
                      ShapeSpan src_shape, ShapeSpan dst_shape,
                      ShapeSpan window_dimensions, ShapeSpan strides,
                      ShapeSpan pad_low) {
  int rank = src_shape.size();
  absl::InlinedVector<int, 8> src_indices(rank, 0);
  absl::InlinedVector<int, 8> dst_indices(rank, 0);
  for (int i = 0, e = GetElementCount(dst_shape); i < e; ++i) {
    for (int j = 0; j < rank; ++j) {
      src_indices[j] = dst_indices[j] * strides[j] - pad_low[j];
    }
    ComputePoolingWindow<T, KernelImpl>(src_buffer, src_indices, src_shape,
                                        init_buffer[0], window_dimensions,
                                        &dst_buffer[i]);
    IncrementShapeIndex(absl::MakeSpan(dst_indices), dst_shape);
  }
  return OkStatus();
}

}  // namespace impl

template <typename T>
Status PoolingSum::Execute(absl::Span<const T> src_buffer,
                           absl::Span<const T> init_buffer,
                           absl::Span<T> dst_buffer, ShapeSpan src_shape,
                           ShapeSpan dst_shape, ShapeSpan window_dimensions,
                           ShapeSpan strides, ShapeSpan pad_low) {
  return impl::GenericPooling<T, impl::SumKernel>(
      src_buffer, init_buffer, dst_buffer, src_shape, dst_shape,
      window_dimensions, strides, pad_low);
}

template <typename T>
Status PoolingMin::Execute(absl::Span<const T> src_buffer,
                           absl::Span<const T> init_buffer,
                           absl::Span<T> dst_buffer, ShapeSpan src_shape,
                           ShapeSpan dst_shape, ShapeSpan window_dimensions,
                           ShapeSpan strides, ShapeSpan pad_low) {
  return impl::GenericPooling<T, impl::MinKernel>(
      src_buffer, init_buffer, dst_buffer, src_shape, dst_shape,
      window_dimensions, strides, pad_low);
}

template <typename T>
Status PoolingMax::Execute(absl::Span<const T> src_buffer,
                           absl::Span<const T> init_buffer,
                           absl::Span<T> dst_buffer, ShapeSpan src_shape,
                           ShapeSpan dst_shape, ShapeSpan window_dimensions,
                           ShapeSpan strides, ShapeSpan pad_low) {
  return impl::GenericPooling<T, impl::MaxKernel>(
      src_buffer, init_buffer, dst_buffer, src_shape, dst_shape,
      window_dimensions, strides, pad_low);
}

}  // namespace kernels
}  // namespace vmla
}  // namespace hal
}  // namespace iree

#endif  // IREE_HAL_VMLA_OP_KERNELS_GENERIC_H_

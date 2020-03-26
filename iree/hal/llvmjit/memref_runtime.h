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
//

#ifndef IREE_HAL_LLVMJIT_LLVMJIT_MEMREF_RUNTIME_H_
#define IREE_HAL_LLVMJIT_LLVMJIT_MEMREF_RUNTIME_H_

#include <assert.h>

#include <cstdint>
#include <vector>

namespace iree {
namespace hal {
namespace llvmjit {

template <int N>
void dropFront(int64_t arr[N], int64_t *res) {
  for (unsigned i = 1; i < N; ++i) *(res + i - 1) = arr[i];
}

/// StridedMemRef descriptor type with static rank.
template <typename T, int N>
struct StridedMemRefType {
  T *basePtr;
  T *data;
  int64_t offset;
  int64_t sizes[N];
  int64_t strides[N];
  // This operator[] is extremely slow and only for sugaring purposes.
  StridedMemRefType<T, N - 1> operator[](int64_t idx) {
    StridedMemRefType<T, N - 1> res;
    res.basePtr = basePtr;
    res.data = data;
    res.offset = offset + idx * strides[0];
    dropFront<N>(sizes, res.sizes);
    dropFront<N>(strides, res.strides);
    return res;
  }
};

/// StridedMemRef descriptor type specialized for rank 1.
template <typename T>
struct StridedMemRefType<T, 1> {
  T *basePtr;
  T *data;
  int64_t offset;
  int64_t sizes[1];
  int64_t strides[1];
  T &operator[](int64_t idx) { return *(data + offset + idx * strides[0]); }
};

/// StridedMemRef descriptor type specialized for rank 0.
template <typename T>
struct StridedMemRefType<T, 0> {
  T *basePtr;
  T *data;
  int64_t offset;
};

// Unranked MemRef
template <typename T>
struct UnrankedMemRefType {
  int64_t rank;
  void *descriptor;
};

// Given a shape with sizes greater than 0 along all dimensions,
// returns the distance, in number of elements, between a slice in a dimension
// and the next slice in the same dimension.
//   e.g. shape[3, 4, 5] -> strides[20, 5, 1]
inline std::vector<int64_t> makeStrides(const std::vector<int64_t> &shape) {
  std::vector<int64_t> tmp;
  if (shape.empty()) return tmp;
  tmp.reserve(shape.size());
  int64_t running = 1;
  for (auto rit = shape.rbegin(), reit = shape.rend(); rit != reit; ++rit) {
    assert(*rit > 0 &&
           "size must be greater than 0 along all dimensions of shape");
    tmp.push_back(running);
    running *= *rit;
  }
  return std::vector<int64_t>(tmp.rbegin(), tmp.rend());
}

// Mallocs a StridedMemRefDescriptor<T, N>* that matches the MLIR ABI.
// This is an implementation detail that is kept in sync with MLIR codegen
// conventions.
template <typename T, int N>
StridedMemRefType<T, N> *makeStridedMemRefDescriptor(
    void *ptr, const std::vector<int64_t> &shape) {
  StridedMemRefType<T, N> *descriptor = static_cast<StridedMemRefType<T, N> *>(
      malloc(sizeof(StridedMemRefType<T, N>)));
  descriptor->basePtr = static_cast<T *>(ptr);
  descriptor->data = static_cast<T *>(ptr);
  descriptor->offset = 0;
  std::copy(shape.begin(), shape.end(), descriptor->sizes);
  auto strides = makeStrides(shape);
  std::copy(strides.begin(), strides.end(), descriptor->strides);
  return descriptor;
}

// Mallocs a StridedMemRefDescriptor<T, 0>* (i.e. a pointer to scalar) that
// matches the MLIR ABI. This is an implementation detail that is kept in sync
// with MLIR codegen conventions.
template <typename T>
StridedMemRefType<T, 0> *makeStridedMemRefDescriptor(
    void *ptr, const std::vector<int64_t> &shape) {
  StridedMemRefType<T, 0> *descriptor = static_cast<StridedMemRefType<T, 0> *>(
      malloc(sizeof(StridedMemRefType<T, 0>)));
  descriptor->basePtr = static_cast<T *>(ptr);
  descriptor->data = static_cast<T *>(ptr);
  descriptor->offset = 0;
  return descriptor;
}

// Mallocs an UnrankedMemRefType<T>* that contains a ranked
// StridedMemRefDescriptor<T, Rank>* and matches the MLIR ABI. This is an
// implementation detail that is kept in sync with MLIR codegen conventions.
template <typename T>
UnrankedMemRefType<T> *allocUnrankedDescriptor(
    void *data, const std::vector<int64_t> &shape) {
  UnrankedMemRefType<T> *res = static_cast<UnrankedMemRefType<T> *>(
      malloc(sizeof(UnrankedMemRefType<T>)));
  res->rank = shape.size();
  if (res->rank == 0)
    res->descriptor = makeStridedMemRefDescriptor<T>(data, shape);
  else if (res->rank == 1)
    res->descriptor = makeStridedMemRefDescriptor<T, 1>(data, shape);
  else if (res->rank == 2)
    res->descriptor = makeStridedMemRefDescriptor<T, 2>(data, shape);
  else if (res->rank == 3)
    res->descriptor = makeStridedMemRefDescriptor<T, 3>(data, shape);
  else if (res->rank == 4)
    res->descriptor = makeStridedMemRefDescriptor<T, 4>(data, shape);
  else if (res->rank == 5)
    res->descriptor = makeStridedMemRefDescriptor<T, 5>(data, shape);
  else if (res->rank == 6)
    res->descriptor = makeStridedMemRefDescriptor<T, 6>(data, shape);
  else
    assert(false && "Unsupported 6+D memref descriptor");
  return res;
}

// Shape and strides aren't used in the generated code (yet).
// TODO(ataei): Delete this version once we can pass shapes.
template <typename T>
UnrankedMemRefType<T> *allocUnrankedDescriptor(void *data) {
  UnrankedMemRefType<T> *res = static_cast<UnrankedMemRefType<T> *>(
      malloc(sizeof(UnrankedMemRefType<T>)));
  res->descriptor = makeStridedMemRefDescriptor<T>(data, {});
  return res;
}

// Frees an UnrankedMemRefType<T>*
template <typename T>
void freeUnrankedDescriptor(UnrankedMemRefType<T> *desc) {
  free(desc->descriptor);
  free(desc);
}

}  // namespace llvmjit
}  // namespace hal
}  // namespace iree
#endif  // IREE_HAL_LLVMJIT_LLVMJIT_MEMREF_RUNTIME_H_

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

#ifndef IREE_HAL_DYLIB_MEMREF_RUNTIME_H_
#define IREE_HAL_DYLIB_MEMREF_RUNTIME_H_

#include <assert.h>

#include <cstdint>
#include <vector>

namespace iree {
namespace hal {
namespace dylib {

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

}  // namespace dylib
}  // namespace hal
}  // namespace iree

#endif  // IREE_HAL_DYLIB_MEMREF_RUNTIME_H_

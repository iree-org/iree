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

// MemRefUtils.h
// -----------------------------------------------------------------------------
//
// Utils for MLIR ABI interfacing with frameworks
//
// The templated free functions below make it possible to allocate dense
// contiguous buffers with shapes that interoperate properly with the MLIR
// codegen ABI.
//
// ```
//  // 1. Compile and build a model, prepare the runner.
//  ModelRunner runner = ...;
//
//  // 2. Allocate managed input and outputs with proper shapes and init value.
//  auto inputLinearInit = [](unsigned idx, float *ptr) { *ptr = 0.032460f; };
//  ManagedUnrankedMemRefDescriptor inputBuffer =
//      makeInitializedUnrankedDescriptor<float>({B, W0}, inputLinearInit);
//  auto outputLinearInit = [](unsigned idx, float *ptr) { *ptr = 0.0f; };
//  ManagedUnrankedMemRefDescriptor outputBuffer =
//      makeInitializedUnrankedDescriptor<float>({B, W3}, outputLinearInit);
//
//  // 3. Pack pointers to MLIR ABI compliant buffers and call the named func.
//  void *packedArgs[2] = {&inputBuffer->descriptor, &outputBuffer->descriptor};
//  runner.engine->invoke(funcName, llvm::MutableArrayRef<void *>{packedArgs});
// ```

#include <memory>
#include <vector>

#include "experimental/ModelBuilder/MLIRRunnerUtils.h"

#ifndef IREE_EXPERIMENTAL_MODELBUILDER_MEMREFUTILS_H_
#define IREE_EXPERIMENTAL_MODELBUILDER_MEMREFUTILS_H_

namespace mlir {
namespace detail {

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
::UnrankedMemRefType<T> *allocUnrankedDescriptor(
    void *data, const std::vector<int64_t> &shape) {
  ::UnrankedMemRefType<T> *res = static_cast<::UnrankedMemRefType<T> *>(
      malloc(sizeof(::UnrankedMemRefType<T>)));
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

// Frees an UnrankedMemRefType<T>*
template <typename T>
void freeUnrankedDescriptor(::UnrankedMemRefType<T> *desc) {
  free(desc->descriptor);
  free(desc);
}

}  // namespace detail

using ManagedUnrankedMemRefDescriptor =
    std::unique_ptr<::UnrankedMemRefType<float>,
                    decltype(&detail::freeUnrankedDescriptor<float>)>;

// Inefficient initializer called on each element during
// `makeInitializedUnrankedDescriptor`. Takes the linear index and the shape so
// that it can work in a generic fashion. The user can capture the shape and
// delinearize if appropriate.
template <typename T>
using LinearInitializer = std::function<void(unsigned idx, T *ptr)>;

// Entry point to allocate a dense buffer with a given `shape` and initializer
// of type PointwiseInitializer.
template <typename T>
ManagedUnrankedMemRefDescriptor makeInitializedUnrankedDescriptor(
    const std::vector<int64_t> &shape, LinearInitializer<T> init) {
  int64_t size = 1;
  for (int64_t s : shape) size *= s;
  auto *data = static_cast<T *>(malloc(size * sizeof(T)));
  for (unsigned i = 0; i < size; ++i) init(i, data + i);
  return ManagedUnrankedMemRefDescriptor(
      detail::allocUnrankedDescriptor<T>(data, shape),
      &detail::freeUnrankedDescriptor);
}

}  // namespace mlir

#endif  // IREE_EXPERIMENTAL_MODELBUILDER_MEMREFUTILS_H_

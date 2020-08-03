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
//  auto inputBuffer = makeInitializedStridedMemRefDescriptor<float, 2>(
//     {B, W0}, inputLinearInit);
//  auto outputLinearInit = [](unsigned idx, float *ptr) { *ptr = 0.0f; };
//  auto outputBuffer = makeInitializedStridedMemRefDescriptor<float, 2>(
//     {B, W3}, outputLinearInit);
//
//  // 3. Pack pointers to MLIR ABI compliant buffers and call the named func.
//  void *packedArgs[2] = {&inputBuffer->descriptor, &outputBuffer->descriptor};
//  runner.engine->invoke(funcName, llvm::MutableArrayRef<void *>{packedArgs});
// ```

#include <algorithm>
#include <array>
#include <cassert>
#include <functional>
#include <memory>

#include "llvm/ADT/Optional.h"
#include "mlir/ExecutionEngine/CRunnerUtils.h"

#ifndef IREE_EXPERIMENTAL_MODELBUILDER_MEMREFUTILS_H_
#define IREE_EXPERIMENTAL_MODELBUILDER_MEMREFUTILS_H_

namespace mlir {
using AllocFunType = std::function<void *(size_t)>;

namespace detail {

// Given a shape with sizes greater than 0 along all dimensions,
// returns the distance, in number of elements, between a slice in a dimension
// and the next slice in the same dimension.
//   e.g. shape[3, 4, 5] -> strides[20, 5, 1]
template <size_t N>
inline std::array<int64_t, N> makeStrides(const std::array<int64_t, N> &shape) {
  if (N == 0) return shape;
  std::array<int64_t, N> res;
  int64_t running = 1;
  for (int64_t idx = N - 1; idx >= 0; --idx) {
    assert(shape[idx] && "size must be nonnegatice for all shape dimensions");
    res[idx] = running;
    running *= shape[idx];
  }
  return res;
}

// Mallocs a StridedMemRefDescriptor<T, N>* that matches the MLIR ABI.
// This is an implementation detail that is kept in sync with MLIR codegen
// conventions.  Additionally takes a `shapeAlloc` array which
// is used instead of `shape` to allocate "more aligned" data and compute the
// corresponding strides.
template <typename T, int N>
typename std::enable_if<(N >= 1), StridedMemRefType<T, N> *>::type
makeStridedMemRefDescriptor(void *ptr, void *alignedPtr,
                            const std::array<int64_t, N> &shape,
                            const std::array<int64_t, N> &shapeAlloc,
                            AllocFunType allocFun = &::malloc) {
  StridedMemRefType<T, N> *descriptor = static_cast<StridedMemRefType<T, N> *>(
      allocFun(sizeof(StridedMemRefType<T, N>)));
  descriptor->basePtr = static_cast<T *>(ptr);
  descriptor->data = static_cast<T *>(alignedPtr);
  descriptor->offset = 0;
  std::copy(shape.begin(), shape.end(), descriptor->sizes);
  auto strides = makeStrides<N>(shapeAlloc);
  std::copy(strides.begin(), strides.end(), descriptor->strides);
  return descriptor;
}

// Mallocs a StridedMemRefDescriptor<T, 0>* that matches the MLIR ABI.
// This is an implementation detail that is kept in sync with MLIR codegen
// conventions.  Additionally takes a `shapeAlloc` array which
// is used instead of `shape` to allocate "more aligned" data and compute the
// corresponding strides.
template <typename T, int N>
typename std::enable_if<(N == 0), StridedMemRefType<T, 0> *>::type
makeStridedMemRefDescriptor(void *ptr, void *alignedPtr,
                            const std::array<int64_t, N> &shape = {},
                            const std::array<int64_t, N> &shapeAlloc = {},
                            AllocFunType allocFun = &::malloc) {
  StridedMemRefType<T, 0> *descriptor = static_cast<StridedMemRefType<T, 0> *>(
      allocFun(sizeof(StridedMemRefType<T, 0>)));
  descriptor->basePtr = static_cast<T *>(ptr);
  descriptor->data = static_cast<T *>(alignedPtr);
  descriptor->offset = 0;
  return descriptor;
}

// Mallocs a StridedMemRefDescriptor<T, N>* that matches the MLIR ABI.
// This is an implementation detail that is kept in sync with MLIR codegen
// conventions.
template <typename T, int N>
typename std::enable_if<(N >= 1), StridedMemRefType<T, N> *>::type
makeStridedMemRefDescriptor(void *ptr, void *alignedPtr,
                            const std::array<int64_t, N> &shape,
                            AllocFunType allocFun = &::malloc) {
  return makeStridedMemRefDescriptor<T, N>(ptr, alignedPtr, shape, shape,
                                           allocFun);
}

// Fixes compilation failures that started after merging cl/324296127.
// // Mallocs a StridedMemRefDescriptor<T, 0>* (i.e. a pointer to scalar) that
// // matches the MLIR ABI. This is an implementation detail that is kept in
// sync
// // with MLIR codegen conventions.
// template <typename T, int N>
// typename std::enable_if<(N == 0), StridedMemRefType<T, 0> *>::type
// makeStridedMemRefDescriptor(void *ptr, void *alignedPtr,
//                             const std::array<int64_t, N> &shape = {},
//                             AllocFunType allocFun = &::malloc) {
//   return makeStridedMemRefDescriptor<T, N>(ptr, alignedPtr, shape, shape,
//                                            allocFun);
// }

// Mallocs an UnrankedMemRefType<T>* that contains a ranked
// StridedMemRefDescriptor<T, Rank>* and matches the MLIR ABI. This is an
// implementation detail that is kept in sync with MLIR codegen conventions.
template <typename T, int N>
::UnrankedMemRefType<T> *allocUnrankedDescriptor(
    void *data, void *alignedData, const std::array<int64_t, N> &shape,
    AllocFunType allocFun = &::malloc) {
  ::UnrankedMemRefType<T> *res = static_cast<::UnrankedMemRefType<T> *>(
      allocFun(sizeof(::UnrankedMemRefType<T>)));
  res->rank = N;
  res->descriptor = makeStridedMemRefDescriptor<T, N>(data, alignedData, shape);
  return res;
}

// Frees an UnrankedMemRefType<T>*
template <typename T>
void freeUnrankedDescriptor(::UnrankedMemRefType<T> *desc) {
  free(desc->descriptor);
  free(desc);
}

}  // namespace detail

//===----------------------------------------------------------------------===//
// Public API
//===----------------------------------------------------------------------===//

// Inefficient initializer called on each element during
// `makeInitializedUnrankedDescriptor`. Takes the linear index and the shape so
// that it can work in a generic fashion. The user can capture the shape and
// delinearize if appropriate.
template <typename T>
using LinearInitializer = std::function<void(unsigned idx, T *ptr)>;

inline uint32_t pow2msb(uint32_t val) {
  assert(val > 0);
  val--;
  val |= val >> 1;
  val |= val >> 2;
  val |= val >> 4;
  val |= val >> 8;
  val |= val >> 16;
  return val + 1;
}

// No such thing as a portable posize_memalign, roll our own.
// [alignment] allow to specify an arbitrary alignment. It must be a power of 2
// and greater than the size of T. By default the alignment is sizeof(T).
template <typename T>
std::pair<void *, void *> allocAligned(
    size_t nElements, AllocFunType allocFun = &::malloc,
    llvm::Optional<uint64_t> alignment = llvm::Optional<uint64_t>()) {
  assert(sizeof(T) < (1ul << 32) && "Elemental type overflows");
  auto size = nElements * sizeof(T);
  auto desiredAlignment = alignment.getValueOr(pow2msb(sizeof(T)));
  assert((desiredAlignment & (desiredAlignment - 1)) == 0);
  assert(desiredAlignment >= sizeof(T));
  void *data = allocFun(size + desiredAlignment);
  uintptr_t addr = reinterpret_cast<uintptr_t>(data);
  uintptr_t rem = addr % desiredAlignment;
  void *alignedData =
      (rem == 0) ? data
                 : reinterpret_cast<void *>(addr + (desiredAlignment - rem));
  assert(reinterpret_cast<uintptr_t>(alignedData) % desiredAlignment == 0);
  return std::pair<void *, void *>(data, alignedData);
}

// Entry point to allocate a dense buffer with a given `shape` and initializer
// of type PointwiseInitializer. Can optionally take specific `alloc` and `free`
// functions.
template <typename T, int N, typename FreeFunType = decltype(&::free)>
std::unique_ptr<::UnrankedMemRefType<float>, FreeFunType>
makeInitializedUnrankedDescriptor(
    const std::array<int64_t, N> &shape, LinearInitializer<T> init,
    llvm::Optional<uint64_t> alignment = llvm::Optional<uint64_t>(),
    AllocFunType alloc = &::malloc, FreeFunType freeFun = &::free) {
  int64_t nElements = 1;
  for (int64_t s : shape) nElements *= s;
  auto allocated = allocAligned<T>(nElements, alloc, alignment);
  auto *data = static_cast<T *>(allocated.first);
  auto *alignedData = static_cast<T *>(allocated.second);
  for (unsigned i = 0; i < nElements; ++i) init(i, alignedData);
  return std::unique_ptr<::UnrankedMemRefType<float>, FreeFunType>(
      detail::allocUnrankedDescriptor<T, N>(data, alignedData, shape), freeFun);
}

// Entry point to allocate a dense buffer with a given `shape` and initializer
// of type PointwiseInitializer. Additionally takes a `shapeAlloc` array which
// is used instead of `shape` to allocate "more aligned" data and compute the
// corresponding strides.
// Can optionally take specific alloc and free functions.
//
// Example:
// When called with `shape = [128, 127]` and `shapeAlloc = [128, 128]`, this
// allocates a memref with `128*128*sizeof(T)` bytes, `sizes = [128, 127]` and
// `strides = [128, 1]`.
template <typename T, int N, typename FreeFunType = decltype(&::free)>
std::unique_ptr<StridedMemRefType<T, N>, FreeFunType>
makeInitializedStridedMemRefDescriptor(
    const std::array<int64_t, N> &shape,
    const std::array<int64_t, N> &shapeAlloc, LinearInitializer<T> init,
    llvm::Optional<uint64_t> alignment = llvm::Optional<uint64_t>(),
    AllocFunType allocFun = &::malloc, FreeFunType freeFun = &::free) {
  for (unsigned i = 0; i < N; ++i)
    assert(shape[i] <= shapeAlloc[i] &&
           "shapeAlloc must be greater than or equal to shape");
  int64_t nElements = 1;
  for (int64_t s : shapeAlloc) nElements *= s;
  auto allocated = allocAligned<T>(nElements, allocFun, alignment);
  auto *data = static_cast<T *>(allocated.first);
  auto *alignedData = static_cast<T *>(allocated.second);
  for (unsigned i = 0; i < nElements; ++i) init(i, alignedData);
  return std::unique_ptr<StridedMemRefType<T, N>, FreeFunType>(
      detail::makeStridedMemRefDescriptor<T, N>(data, alignedData, shape,
                                                shapeAlloc, allocFun),
      freeFun);
}

// Entry point to allocate a dense buffer with a given `shape` and initializer
// of type PointwiseInitializer. Can optionally take specific alloc and free
// functions.
template <typename T, int N, typename FreeFunType = decltype(&::free)>
std::unique_ptr<StridedMemRefType<T, N>, FreeFunType>
makeInitializedStridedMemRefDescriptor(
    const std::array<int64_t, N> &shape, LinearInitializer<T> init,
    llvm::Optional<uint64_t> alignment = llvm::Optional<uint64_t>(),
    AllocFunType allocFun = &::malloc, FreeFunType freeFun = &::free) {
  return makeInitializedStridedMemRefDescriptor<T, N>(
      shape, shape, init, alignment, allocFun, freeFun);
}

}  // namespace mlir

#endif  // IREE_EXPERIMENTAL_MODELBUILDER_MEMREFUTILS_H_

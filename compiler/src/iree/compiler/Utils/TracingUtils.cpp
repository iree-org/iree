// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Utils/TracingUtils.h"

namespace mlir {
namespace iree_compiler {

#if IREE_ENABLE_COMPILER_TRACING && \
    IREE_TRACING_FEATURES & IREE_TRACING_FEATURE_INSTRUMENTATION

namespace {
thread_local llvm::SmallVector<iree_zone_id_t, 8> passTraceZonesStack;
}  // namespace

void PassTracing::runBeforePass(Pass *pass, Operation *op) {
  IREE_TRACE_ZONE_BEGIN_EXTERNAL(z0, __FILE__, strlen(__FILE__), __LINE__,
                                 pass->getName().data(), pass->getName().size(),
                                 NULL, 0);
  passTraceZonesStack.push_back(z0);
}
void PassTracing::runAfterPass(Pass *pass, Operation *op) {
  IREE_TRACE_ZONE_END(passTraceZonesStack.back());
  passTraceZonesStack.pop_back();
}
void PassTracing::runAfterPassFailed(Pass *pass, Operation *op) {
  IREE_TRACE_ZONE_END(passTraceZonesStack.back());
  passTraceZonesStack.pop_back();
}

#endif  // IREE_TRACING_FEATURES & IREE_TRACING_FEATURE_INSTRUMENTATION

}  // namespace iree_compiler
}  // namespace mlir

#if IREE_TRACING_FEATURES & IREE_TRACING_FEATURE_ALLOCATION_TRACKING

// Replace most forms of the allocation and deallocation functions.

void *operator new(std::size_t count) {
  auto ptr = malloc(count);
  IREE_TRACE_ALLOC(ptr, count);
  return ptr;
}
void *operator new[](std::size_t count) {
  auto ptr = malloc(count);
  IREE_TRACE_ALLOC(ptr, count);
  return ptr;
}
void *operator new(std::size_t count, std::align_val_t al) {
  auto ptr = malloc(count);
  IREE_TRACE_ALLOC(ptr, count);
  return ptr;
}
void *operator new[](std::size_t count, std::align_val_t al) {
  auto ptr = malloc(count);
  IREE_TRACE_ALLOC(ptr, count);
  return ptr;
}

void operator delete(void *ptr) noexcept {
  IREE_TRACE_FREE(ptr);
  free(ptr);
}
void operator delete[](void *ptr) noexcept {
  IREE_TRACE_FREE(ptr);
  free(ptr);
}
void operator delete(void *ptr, size_t sz) noexcept {
  IREE_TRACE_FREE(ptr);
  free(ptr);
}
void operator delete[](void *ptr, size_t sz) noexcept {
  IREE_TRACE_FREE(ptr);
  free(ptr);
}
void operator delete(void *ptr, std::align_val_t al) noexcept {
  IREE_TRACE_FREE(ptr);
  free(ptr);
}
void operator delete[](void *ptr, std::align_val_t al) noexcept {
  IREE_TRACE_FREE(ptr);
  free(ptr);
}
void operator delete(void *ptr, size_t sz, std::align_val_t al) noexcept {
  IREE_TRACE_FREE(ptr);
  free(ptr);
}
void operator delete[](void *ptr, size_t sz, std::align_val_t al) noexcept {
  IREE_TRACE_FREE(ptr);
  free(ptr);
}

#endif  // IREE_TRACING_FEATURE_ALLOCATION_TRACKING

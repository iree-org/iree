// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Utils/TracingUtils.h"

#if IREE_ENABLE_COMPILER_TRACING &&                                            \
    IREE_TRACING_FEATURES & IREE_TRACING_FEATURE_INSTRUMENTATION

#include "mlir/IR/SymbolTable.h"
#include "mlir/Pass/PassManager.h"

namespace mlir::iree_compiler {

//===----------------------------------------------------------------------===//
// PassTracing (PassInstrumentation)
//===----------------------------------------------------------------------===//

namespace {
thread_local llvm::SmallVector<iree_zone_id_t, 8> passTraceZonesStack;
} // namespace

static void prettyPrintOpBreadcrumb(Operation *op, llvm::raw_ostream &os) {
  auto parentOp = op->getParentOp();
  if (parentOp) {
    prettyPrintOpBreadcrumb(parentOp, os);
    os << " > ";
  }
  os << op->getName();
  if (auto symbolOp = dyn_cast<SymbolOpInterface>(op)) {
    os << " @" << symbolOp.getName();
  }
}

void PassTracing::runBeforePass(Pass *pass, Operation *op) {
  IREE_TRACE_ZONE_BEGIN_EXTERNAL(z0, __FILE__, strlen(__FILE__), __LINE__,
                                 pass->getName().data(), pass->getName().size(),
                                 NULL, 0);
  passTraceZonesStack.push_back(z0);

  std::string breadcrumbStorage;
  llvm::raw_string_ostream os(breadcrumbStorage);
  prettyPrintOpBreadcrumb(op, os);
  IREE_TRACE_ZONE_APPEND_TEXT(z0, os.str().data());
}
void PassTracing::runAfterPass(Pass *pass, Operation *op) {
  IREE_TRACE_ZONE_END(passTraceZonesStack.back());
  passTraceZonesStack.pop_back();
}
void PassTracing::runAfterPassFailed(Pass *pass, Operation *op) {
  IREE_TRACE_ZONE_END(passTraceZonesStack.back());
  passTraceZonesStack.pop_back();
}

//===----------------------------------------------------------------------===//
// MarkBeginPass / MarkEndPass
//===----------------------------------------------------------------------===//

namespace {

class TraceFrameMarkBeginPass
    : public PassWrapper<TraceFrameMarkBeginPass, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TraceFrameMarkBeginPass);

  TraceFrameMarkBeginPass() = default;
  TraceFrameMarkBeginPass(llvm::StringRef name) { this->name = name; }

  void runOnOperation() override {
    if (!name.empty()) {
      IREE_TRACE_FRAME_MARK(); // Top level (unnamed) frame.
      IREE_TRACE_FRAME_MARK_BEGIN_NAMED(name.data());
    }
  }

  llvm::StringRef name;
};

class TraceFrameMarkEndPass
    : public PassWrapper<TraceFrameMarkEndPass, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TraceFrameMarkEndPass);

  TraceFrameMarkEndPass() = default;
  TraceFrameMarkEndPass(llvm::StringRef name) { this->name = name; }

  void runOnOperation() override {
    if (!name.empty()) {
      IREE_TRACE_FRAME_MARK_END_NAMED(name.data());
    }
  }

  llvm::StringRef name;
};

} // namespace

std::unique_ptr<OperationPass<ModuleOp>>
createTraceFrameMarkBeginPass(llvm::StringRef name) {
  return std::make_unique<TraceFrameMarkBeginPass>(name);
}

std::unique_ptr<OperationPass<ModuleOp>>
createTraceFrameMarkEndPass(llvm::StringRef name) {
  return std::make_unique<TraceFrameMarkEndPass>(name);
}

} // namespace mlir::iree_compiler

//===----------------------------------------------------------------------===//
// Allocation tracking
//===----------------------------------------------------------------------===//

#if IREE_TRACING_FEATURES & IREE_TRACING_FEATURE_ALLOCATION_TRACKING

// Mark memory events by overloading `operator new` and `operator delete` and
// using the `IREE_TRACE_ALLOC` and `IREE_TRACE_FREE` annotations.
//
// The `new` and `delete` operators are designed to be replaceable, though this
// is a very large and brittle hammer:
// https://en.cppreference.com/w/cpp/memory/new/operator_new
// https://en.cppreference.com/w/cpp/memory/new/operator_delete
//   * "Versions (1-8) are replaceable: a user-provided non-member function
//      with the same signature defined anywhere in the program, in any source
//      file, replaces the default version"
//   * "The program is ill-formed, no diagnostic required if more than one
//      replacement is provided in the program for any of the replaceable
//      allocation function."
//
// We should also still honor alignment:
// https://www.cppstories.com/2019/08/newnew-align/#custom-overloads
// Note: always disable exceptions, so no `if (!ptr) throw std::bad_alloc{};`

// Avoid potential sharp edge by making allocation tracking and sanitizers
// mutually exclusive. They _might_ work together, but here's a warning anyway.
#if defined(__has_feature)
#if __has_feature(address_sanitizer) || __has_feature(memory_sanitizer) ||     \
    __has_feature(thread_sanitizer)
#error Compiler IREE_TRACING_FEATURE_ALLOCATION_TRACKING not compatible with sanitizers
#endif // __has_feature(*_sanitizer)
#endif // defined(__has_feature)

#include <new>

void *iree_aligned_new(std::size_t count, std::align_val_t al) {
#if defined(_WIN32) || defined(__CYGWIN__)
  return _aligned_malloc(count, static_cast<std::size_t>(al));
#else
  return aligned_alloc(static_cast<std::size_t>(al), count);
#endif
}

// replaceable allocation functions
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
  auto ptr = iree_aligned_new(count, al);
  IREE_TRACE_ALLOC(ptr, count);
  return ptr;
}
void *operator new[](std::size_t count, std::align_val_t al) {
  auto ptr = iree_aligned_new(count, al);
  IREE_TRACE_ALLOC(ptr, count);
  return ptr;
}

// replaceable non-throwing allocation functions
// (even though we disable exceptions, these have unique signatures)
void *operator new(std::size_t count, const std::nothrow_t &tag) noexcept {
  auto ptr = malloc(count);
  IREE_TRACE_ALLOC(ptr, count);
  return ptr;
}
void *operator new[](std::size_t count, const std::nothrow_t &tag) noexcept {
  auto ptr = malloc(count);
  IREE_TRACE_ALLOC(ptr, count);
  return ptr;
}
void *operator new(std::size_t count, std::align_val_t al,
                   const std::nothrow_t &) noexcept {
  auto ptr = iree_aligned_new(count, al);
  IREE_TRACE_ALLOC(ptr, count);
  return ptr;
}
void *operator new[](std::size_t count, std::align_val_t al,
                     const std::nothrow_t &) noexcept {
  auto ptr = iree_aligned_new(count, al);
  IREE_TRACE_ALLOC(ptr, count);
  return ptr;
}

void iree_aligned_free(void *ptr) {
#if defined(_WIN32) || defined(__CYGWIN__)
  _aligned_free(ptr);
#else
  free(ptr);
#endif
}

// replaceable usual deallocation functions
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
  iree_aligned_free(ptr);
}
void operator delete[](void *ptr, std::align_val_t al) noexcept {
  IREE_TRACE_FREE(ptr);
  iree_aligned_free(ptr);
}
void operator delete(void *ptr, size_t sz, std::align_val_t al) noexcept {
  IREE_TRACE_FREE(ptr);
  iree_aligned_free(ptr);
}
void operator delete[](void *ptr, size_t sz, std::align_val_t al) noexcept {
  IREE_TRACE_FREE(ptr);
  iree_aligned_free(ptr);
}

#endif // IREE_TRACING_FEATURE_ALLOCATION_TRACKING

#endif // IREE_ENABLE_COMPILER_TRACING + IREE_TRACING_FEATURE_INSTRUMENTATION

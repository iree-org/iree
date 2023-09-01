// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_BASE_INTERNAL_DEBUGGING_H_
#define IREE_BASE_INTERNAL_DEBUGGING_H_

#include "iree/base/target_platform.h"

#ifdef __cplusplus
extern "C" {
#endif

#if defined(IREE_COMPILER_GCC_COMPAT)
#define IREE_ATTRIBUTE_ALWAYS_INLINE __attribute__((always_inline))
#elif defined(IREE_COMPILER_MSVC)
#define IREE_ATTRIBUTE_ALWAYS_INLINE __forceinline
#else
#define IREE_ATTRIBUTE_ALWAYS_INLINE
#endif  // IREE_COMPILER_*

//===----------------------------------------------------------------------===//
// Debugger interaction
//===----------------------------------------------------------------------===//
// NOTE: in general it's not a good idea to change program behavior when running
// under a debugger as that then makes it harder to reproduce and successfully
// debug issues that happen without the debugger.

// Forces a break into an attached debugger.
// May be ignored if no debugger is attached or raise a signal that gives the
// option to attach a debugger.
//
// We implement this directly in the header with ALWAYS_INLINE so that the
// stack doesn't get all messed up.
IREE_ATTRIBUTE_ALWAYS_INLINE static inline void iree_debug_break(void) {
#if IREE_HAVE_BUILTIN(__builtin_debugtrap)
  __builtin_debugtrap();
#elif defined(IREE_PLATFORM_WINDOWS)
  __debugbreak();
#elif defined(IREE_ARCH_ARM_32)
  __asm__ volatile(".inst 0xe7f001f0");
#elif defined(IREE_ARCH_ARM_64)
  __asm__ volatile(".inst 0xd4200000");
#elif defined(IREE_ARCH_X86_32) || defined(IREE_ARCH_X86_64)
  __asm__ volatile("int $0x03");
#elif defined(IREE_PLATFORM_EMSCRIPTEN)
  EM_ASM({ debugger; });
#else
  // NOTE: this is unrecoverable and debugging cannot continue.
  __builtin_trap();
#endif  // __builtin_debugtrap
}

//===----------------------------------------------------------------------===//
// Sanitizer interfaces
//===----------------------------------------------------------------------===//
// These provide hints to the various -fsanitize= features that help us indicate
// what our code is doing to prevent false positives and gain additional
// coverage. By default the sanitizers try to hook platform features like
// mutexes and threads and our own implementations of those aren't automatically
// picked up. In addition, specific uses of memory like arenas can thwart tools
// like ASAN that try to detect accesses to freed memory because we are never
// actually malloc()'ing and free()'ing and need to tell ASAN when blocks of
// memory come into/out-of the pool.
//
// The documentation on these interfaces is pretty sparse but it's possible to
// find usage examples of the hooks in the compiler-provided hooks themselves.
//
// The headers can be viewed here:
// https://github.com/llvm/llvm-project/tree/main/compiler-rt/include/sanitizer
// And common interceptors here:
// https://github.com/llvm/llvm-project/blob/main/compiler-rt/lib/tsan/rtl/tsan_interceptors_posix.cpp
//
// NOTE: don't assume the presence of a sanitizer implies clang+llvm+x86! GCC
// supports all of the sanitizers and MSVC supports ASAN and almost all of them
// can be used on non-x86 platforms.

#if defined(IREE_SANITIZER_ADDRESS)
#include <sanitizer/asan_interface.h>
#include <sanitizer/lsan_interface.h>
#endif  // IREE_SANITIZER_ADDRESS

// For whenever we want to provide specialized msan/tsan hooks:
//   #if defined(IREE_SANITIZER_MEMORY)
//   #include <sanitizer/msan_interface.h>
//   #endif  // IREE_SANITIZER_MEMORY
//   #if defined(IREE_SANITIZER_THREAD)
//   #include <sanitizer/tsan_interface.h>
//   #endif  // IREE_SANITIZER_THREAD

// Suppresses leak detection false-positives in a region. May be nested.
// Do not use this for any IREE-owned code: fix your leaks! This is useful when
// third-party libraries or system calls may create false positives or just be
// leaky such as GPU drivers and shader compilers (which are notoriously bad).
#if defined(IREE_SANITIZER_ADDRESS)
#define IREE_LEAK_CHECK_DISABLE_PUSH() __lsan_disable()
#define IREE_LEAK_CHECK_DISABLE_POP() __lsan_enable()
#else
#define IREE_LEAK_CHECK_DISABLE_PUSH()
#define IREE_LEAK_CHECK_DISABLE_POP()
#endif  // IREE_SANITIZER_ADDRESS

// Manual address poisoning; see
// https://github.com/google/sanitizers/wiki/AddressSanitizerManualPoisoning.
#if defined(IREE_SANITIZER_ADDRESS)
// Marks memory region [addr, addr+size) as unaddressable.
// This memory must be previously allocated by the user program. Accessing
// addresses in this region from instrumented code is forbidden until
// this region is unpoisoned. This function is not guaranteed to poison
// the whole region - it may poison only subregion of [addr, addr+size) due
// to ASan alignment restrictions.
// Method is NOT thread-safe in the sense that no two threads can
// (un)poison memory in the same memory region simultaneously.
#define IREE_ASAN_POISON_MEMORY_REGION(addr, size) \
  __asan_poison_memory_region((addr), (size))
// Marks memory region [addr, addr+size) as addressable.
// This memory must be previously allocated by the user program. Accessing
// addresses in this region is allowed until this region is poisoned again.
// This function may unpoison a superregion of [addr, addr+size) due to
// ASan alignment restrictions.
// Method is NOT thread-safe in the sense that no two threads can
// (un)poison memory in the same memory region simultaneously.
#define IREE_ASAN_UNPOISON_MEMORY_REGION(addr, size) \
  __asan_unpoison_memory_region((addr), (size))
#else
#define IREE_ASAN_POISON_MEMORY_REGION(addr, size) ((void)(addr), (void)(size))
#define IREE_ASAN_UNPOISON_MEMORY_REGION(addr, size) \
  ((void)(addr), (void)(size))
#endif  // IREE_SANITIZER_ADDRESS

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // IREE_BASE_INTERNAL_DEBUGGING_H_

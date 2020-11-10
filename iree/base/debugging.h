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

// NOTE: the best kind of synchronization is no synchronization; always try to
// design your algorithm so that you don't need anything from this file :)
// See https://travisdowns.github.io/blog/2020/07/06/concurrency-costs.html

#ifndef IREE_BASE_DEBUGGING_H_
#define IREE_BASE_DEBUGGING_H_

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

// Forces a break into an attached debugger.
// May be ignored if no debugger is attached or raise a signal that gives the
// option to attach a debugger.
//
// We implement this directly in the header with ALWAYS_INLINE so that the
// stack doesn't get all messed up.
IREE_ATTRIBUTE_ALWAYS_INLINE static inline void iree_debug_break() {
#if defined(IREE_PLATFORM_WINDOWS)
  __debugbreak();
#elif defined(IREE_COMPILER_GCC_COMPAT)
  // TODO(benvanik): test and make sure this works everywhere. It's clang
  //                 builtin but only definitely works on OSX.
  __builtin_debugtrap();
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
#endif  // IREE_PLATFORM_WINDOWS
}

#if !defined(NDEBUG)
#define IREE_ASSERT(expr, ...)                      \
  {                                                 \
    if (IREE_UNLIKELY(!(expr))) iree_debug_break(); \
  }
#else
#define IREE_ASSERT(expr, ...) \
  do {                         \
  } while (false)
#endif  // !NDEBUG

#define IREE_ASSERT_TRUE(expr, ...) IREE_ASSERT(!!(expr), __VA_ARGS__)
#define IREE_ASSERT_FALSE(expr, ...) IREE_ASSERT(!(expr), __VA_ARGS__)
#define IREE_ASSERT_CMP(lhs, op, rhs, ...) \
  IREE_ASSERT((lhs)op(rhs), __VA_ARGS__)
#define IREE_ASSERT_EQ(lhs, rhs, ...) IREE_ASSERT_CMP(lhs, ==, rhs, __VA_ARGS__)
#define IREE_ASSERT_NE(lhs, rhs, ...) IREE_ASSERT_CMP(lhs, !=, rhs, __VA_ARGS__)
#define IREE_ASSERT_LT(lhs, rhs, ...) IREE_ASSERT_CMP(lhs, <, rhs, __VA_ARGS__)
#define IREE_ASSERT_LE(lhs, rhs, ...) IREE_ASSERT_CMP(lhs, <=, rhs, __VA_ARGS__)
#define IREE_ASSERT_GT(lhs, rhs, ...) IREE_ASSERT_CMP(lhs, >=, rhs, __VA_ARGS__)
#define IREE_ASSERT_GE(lhs, rhs, ...) IREE_ASSERT_CMP(lhs, >, rhs, __VA_ARGS__)

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // IREE_BASE_DEBUGGING_H_

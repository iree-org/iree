// Copyright 2019 Google LLC
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

#ifndef IREE_BASE_TARGET_PLATFORM_H_
#define IREE_BASE_TARGET_PLATFORM_H_

#include <stdint.h>

// The build system defines one of the following top-level platforms and then
// one platform+architecture pair for that platform.
//
// IREE_ARCH_ARM_32
// IREE_ARCH_ARM_64
// IREE_ARCH_RISCV_32
// IREE_ARCH_RISCV_64
// IREE_ARCH_WASM_32
// IREE_ARCH_WASM_64
// IREE_ARCH_X86_32
// IREE_ARCH_X86_64
//
// IREE_PTR_SIZE
// IREE_PTR_SIZE_32
// IREE_PTR_SIZE_64
//
// IREE_ENDIANNESS_LITTLE
// IREE_ENDIANNESS_BIG
//
// IREE_COMPILER_CLANG
// IREE_COMPILER_GCC
// IREE_COMPILER_GCC_COMPAT
// IREE_COMPILER_MSVC
//
// IREE_SANITIZER_ADDRESS
// IREE_SANITIZER_MEMORY
// IREE_SANITIZER_THREAD
//
// IREE_PLATFORM_ANDROID
// IREE_PLATFORM_ANDROID_EMULATOR
// IREE_PLATFORM_APPLE (IOS | MACOS)
// IREE_PLATFORM_EMSCRIPTEN
// IREE_PLATFORM_GENERIC
// IREE_PLATFORM_IOS
// IREE_PLATFORM_IOS_SIMULATOR
// IREE_PLATFORM_LINUX
// IREE_PLATFORM_MACOS
// IREE_PLATFORM_WINDOWS
//
// The special define IREE_PLATFORM_GOOGLE will be specified if the build
// is being performed within the internal Google repository.

//==============================================================================
// IREE_ARCH_*
//==============================================================================

#if defined(__arm__) || defined(__arm64) || defined(__aarch64__) || \
    defined(__thumb__) || defined(__TARGET_ARCH_ARM) ||             \
    defined(__TARGET_ARCH_THUMB) || defined(_M_ARM)
#if defined(__arm64) || defined(__aarch64__)
#define IREE_ARCH_ARM_64 1
#else
#define IREE_ARCH_ARM_32 1
#endif  // __arm64
#endif  // ARM

#if defined(__wasm32__)
#define IREE_ARCH_WASM_32 1
#elif defined(__wasm64__)
#define IREE_ARCH_WASM_64 1
#endif  // WASM

#if defined(__i386__) || defined(__i486__) || defined(__i586__) || \
    defined(__i686__) || defined(__i386) || defined(_M_IX86) || defined(_X86_)
#define IREE_ARCH_X86_32 1
#elif defined(__x86_64) || defined(__x86_64__) || defined(__amd64__) || \
    defined(__amd64) || defined(_M_X64)
#define IREE_ARCH_X86_64 1
#endif  // X86

#if defined(__riscv) && (__riscv_xlen == 32)
#define IREE_ARCH_RISCV_32 1
#elif defined(__riscv) && (__riscv_xlen == 64)
#define IREE_ARCH_RISCV_64 1
#endif

#if !defined(IREE_ARCH_ARM_32) && !defined(IREE_ARCH_ARM_64) &&     \
    !defined(IREE_ARCH_RISCV_32) && !defined(IREE_ARCH_RISCV_64) && \
    !defined(IREE_ARCH_WASM_32) && !defined(IREE_ARCH_WASM_64) &&   \
    !defined(IREE_ARCH_X86_32) && !defined(IREE_ARCH_X86_64)
#error Unknown architecture.
#endif  // all archs

//==============================================================================
// IREE_PTR_SIZE_*
//==============================================================================

#if UINTPTR_MAX > UINT_MAX
#define IREE_PTR_SIZE_64
#define IREE_PTR_SIZE 8
#else
#define IREE_PTR_SIZE_32
#define IREE_PTR_SIZE 4
#endif

//==============================================================================
// IREE_ENDIANNESS_*
//==============================================================================
// https://en.wikipedia.org/wiki/Endianness

#if (defined(__BYTE_ORDER__) && defined(__ORDER_LITTLE_ENDIAN__) && \
     __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__)
#define IREE_ENDIANNESS_LITTLE 1
#elif defined(__BYTE_ORDER__) && defined(__ORDER_BIG_ENDIAN__) && \
    __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
#define IREE_ENDIANNESS_BIG 1
#elif defined(_WIN32)
#define IREE_ENDIANNESS_LITTLE 1
#else
#error IREE endian detection needs to be set up for your compiler
#endif  // __BYTE_ORDER__

//==============================================================================
// IREE_COMPILER_*
//==============================================================================

#if defined(__clang__)
#define IREE_COMPILER_CLANG 1
#define IREE_COMPILER_GCC_COMPAT 1
#elif defined(__GNUC__)
#define IREE_COMPILER_GCC 1
#define IREE_COMPILER_GCC_COMPAT 1
#elif defined(_MSC_VER)
#define IREE_COMPILER_MSVC 1
#else
#error Unrecognized compiler.
#endif  // compiler versions

#if defined(__has_feature)
#if __has_feature(address_sanitizer)
#define IREE_SANITIZER_ADDRESS 1
#endif  // __has_feature(address_sanitizer)
#if __has_feature(memory_sanitizer)
#define IREE_SANITIZER_MEMORY 1
#endif  // __has_feature(memory_sanitizer)
#if __has_feature(thread_sanitizer)
#define IREE_SANITIZER_THREAD 1
#endif  // __has_feature(thread_sanitizer)
#endif  // defined(__has_feature)

//==============================================================================
// IREE_COMPILER_HAS_BUILTIN_DEBUG_TRAP
//==============================================================================

#if defined __has_builtin
#if __has_builtin(__builtin_debugtrap)
#define IREE_COMPILER_HAS_BUILTIN_DEBUG_TRAP 1
#endif
#endif

//==============================================================================
// IREE_PLATFORM_ANDROID
//==============================================================================

#if defined(__ANDROID__)
#define IREE_PLATFORM_ANDROID 1
#endif  // __ANDROID__

//==============================================================================
// IREE_PLATFORM_EMSCRIPTEN
//==============================================================================

#if defined(__EMSCRIPTEN__)
#define IREE_PLATFORM_EMSCRIPTEN 1
#endif  // __ANDROID__

//==============================================================================
// IREE_PLATFORM_IOS | IREE_PLATFORM_MACOS
//==============================================================================

#if defined(__APPLE__)
#include <TargetConditionals.h>
#if TARGET_OS_IPHONE
#define IREE_PLATFORM_IOS 1
#else
#define IREE_PLATFORM_MACOS 1
#endif  // TARGET_OS_IPHONE
#if TARGET_IPHONE_SIMULATOR
#define IREE_PLATFORM_IOS_SIMULATOR 1
#endif  // TARGET_IPHONE_SIMULATOR
#endif  // __APPLE__

#if defined(IREE_PLATFORM_IOS) || defined(IREE_PLATFORM_MACOS)
#define IREE_PLATFORM_APPLE 1
#endif  // IREE_PLATFORM_IOS || IREE_PLATFORM_MACOS

//==============================================================================
// IREE_PLATFORM_LINUX
//==============================================================================

#if defined(__linux__) || defined(linux) || defined(__linux)
#define IREE_PLATFORM_LINUX 1
#endif  // __linux__

//==============================================================================
// IREE_PLATFORM_WINDOWS
//==============================================================================

#if defined(_WIN32) || defined(_WIN64)
#define IREE_PLATFORM_WINDOWS 1
#endif  // _WIN32 || _WIN64

#if defined(IREE_PLATFORM_WINDOWS)

#if defined(_MSC_VER)
// Abseil compatibility: don't include incompatible winsock versions.
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif  // WIN32_LEAN_AND_MEAN
// Abseil compatibility: don't define min and max macros.
#ifndef NOMINMAX
#define NOMINMAX
#endif  // NOMINMAX
#endif  // _MSC_VER

#include <windows.h>

// WinGDI.h defines `ERROR`, undef to avoid conflict naming.
#undef ERROR

#endif  // IREE_PLATFORM_WINDOWS

//==============================================================================
// Fallthrough for unsupported platforms
//==============================================================================

#if !defined(IREE_PLATFORM_ANDROID) && !defined(IREE_PLATFORM_EMSCRIPTEN) && \
    !defined(IREE_PLATFORM_GENERIC) && !defined(IREE_PLATFORM_IOS) &&        \
    !defined(IREE_PLATFORM_LINUX) && !defined(IREE_PLATFORM_MACOS) &&        \
    !defined(IREE_PLATFORM_WINDOWS)
#error Unknown platform.
#endif  // all archs

#endif  // IREE_BASE_TARGET_PLATFORM_H_

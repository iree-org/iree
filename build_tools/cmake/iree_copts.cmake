# Copyright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

#-------------------------------------------------------------------------------
# Abseil configuration
#-------------------------------------------------------------------------------

include(AbseilConfigureCopts)

# By default Abseil strips string literals on mobile platforms, which means
# we cannot run IREE binaries via command-line with proper options. Turn off
# the stripping.
# TODO(#3814): remove ABSL flags.
if(ANDROID)
  add_definitions(-DABSL_FLAGS_STRIP_NAMES=0)
endif()

#-------------------------------------------------------------------------------
# C/C++ options as used within IREE
#-------------------------------------------------------------------------------
#
#         ██     ██  █████  ██████  ███    ██ ██ ███    ██  ██████
#         ██     ██ ██   ██ ██   ██ ████   ██ ██ ████   ██ ██
#         ██  █  ██ ███████ ██████  ██ ██  ██ ██ ██ ██  ██ ██   ███
#         ██ ███ ██ ██   ██ ██   ██ ██  ██ ██ ██ ██  ██ ██ ██    ██
#          ███ ███  ██   ██ ██   ██ ██   ████ ██ ██   ████  ██████
#
# Everything here is added to *every* iree_cc_library/iree_cc_binary/etc.
# That includes both runtime and compiler components, and these may propagate
# out to user code interacting with either (such as custom modules).
#
# Be extremely judicious in the use of these flags.
#
# - Need to disable a warning?
#   Usually these are encountered in compiler-specific code and can be disabled
#   in a compiler-specific way. Only add global warning disables when it's clear
#   that we never want them or that they'll show up in a lot of places.
#
#   See: https://stackoverflow.com/questions/3378560/how-to-disable-gcc-warnings-for-a-few-lines-of-code
#
# - Need to add a linker dependency?
#   First figure out if you *really* need it. If it's only required on specific
#   platforms and in very specific files clang or msvc are used prefer
#   autolinking. GCC is stubborn and doesn't have autolinking so additional
#   flags may be required there.
#
#   See: https://en.wikipedia.org/wiki/Auto-linking
#
# - Need to tweak a compilation mode setting (debug/asserts/etc)?
#   Don't do that here, and in general *don't do that at all* unless it's behind
#   a very specific IREE-prefixed cmake flag (like IREE_SIZE_OPTIMIZED).
#   There's no one-size solution when we are dealing with cross-project and
#   cross-compiled binaries - there's no safe way to set global options that
#   won't cause someone to break, and you probably don't really need to do
#   change that setting anyway. Follow the rule of least surprise: if the user
#   has CMake's Debug configuration active then don't force things into release
#   mode, etc.
#
# - Need to add an include directory?
#   Don't do that here. Always prefer to fully-specify the path from the IREE
#   workspace root when it's known that the compilation will be occuring using
#   the files within the IREE checkout; for example, instead of adding a global
#   include path to third_party/foo/ and #include <foo.h>'ing, just
#   #include "third_party/foo/foo.h". This reduces build configuration, makes it
#   easier for readers to find the files, etc.
#
# - Still think you need to add an include directory? (system includes, etc)
#   Don't do that here, either. It's highly doubtful that every single target in
#   all of IREE (both compiler and runtime) on all platforms (both host and
#   cross-compilation targets) needs your special include directory. Add it on
#   the COPTS of the target you are using it in and, ideally, private to that
#   target (used in .c/cc files, not in a .h that leaks the include path
#   requirements to all consumers of the API).

set(IREE_CXX_STANDARD ${CMAKE_CXX_STANDARD})

# TODO(benvanik): fix these names (or remove entirely).
set(IREE_ROOT_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(IREE_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(IREE_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR})

iree_select_compiler_opts(IREE_DEFAULT_COPTS
  CLANG
    # Set clang diagnostics. These largely match the set of warnings used within
    # Google. They have not been audited super carefully by the IREE team but
    # are generally thought to be a good set and consistency with those used
    # internally is very useful when importing. If you feel that some of these
    # should be different, please raise an issue!

    # Please keep these in sync with build_tools/bazel/iree.bazelrc

    "-Wall"

    # Disable warnings we don't care about or that generally have a low
    # signal/noise ratio.
    "-Wno-ambiguous-member-template"
    "-Wno-char-subscripts"
    "-Wno-deprecated-declarations"
    "-Wno-extern-c-compat" # Matches upstream. Cannot impact due to extern C inclusion method.
    "-Wno-gnu-alignof-expression"
    "-Wno-gnu-variable-sized-type-not-at-end"
    "-Wno-ignored-optimization-argument"
    "-Wno-invalid-offsetof" # Technically UB but needed for intrusive ptrs
    "-Wno-invalid-source-encoding"
    "-Wno-mismatched-tags"
    "-Wno-pointer-sign"
    "-Wno-reserved-user-defined-literal"
    "-Wno-return-type-c-linkage"
    "-Wno-self-assign-overloaded"
    "-Wno-sign-compare"
    "-Wno-signed-unsigned-wchar"
    "-Wno-strict-overflow"
    "-Wno-trigraphs"
    "-Wno-unknown-pragmas"
    "-Wno-unknown-warning-option"
    "-Wno-unused-command-line-argument"
    "-Wno-unused-const-variable"
    "-Wno-unused-function"
    "-Wno-unused-local-typedef"
    "-Wno-unused-private-field"
    "-Wno-user-defined-warnings"
    # Explicitly enable some additional warnings.
    # Some of these aren't on by default, or under -Wall, or are subsets of
    # warnings turned off above.
    "-Wno-ambiguous-member-template"
    "-Wctad-maybe-unsupported"
    "-Wfloat-overflow-conversion"
    "-Wfloat-zero-conversion"
    "-Wfor-loop-analysis"
    "-Wformat-security"
    "-Wgnu-redeclared-enum"
    "-Wimplicit-fallthrough"
    "-Winfinite-recursion"
    "-Wliteral-conversion"
    "-Wnon-virtual-dtor"
    "-Woverloaded-virtual"
    "-Wself-assign"
    "-Wstring-conversion"
    "-Wtautological-overlap-compare"
    "-Wthread-safety"
    "-Wthread-safety-beta"
    "-Wunused-comparison"
    "-Wvla"

    # Turn off some additional warnings (CMake only)
    "-Wno-strict-prototypes"
    "-Wno-shadow-uncaptured-local"
    "-Wno-gnu-zero-variadic-macro-arguments"
    "-Wno-shadow-field-in-constructor"
    "-Wno-unreachable-code-return"
    "-Wno-missing-variable-declarations"
    "-Wno-gnu-label-as-value"
  CLANG_OR_GCC
    "-Wno-unused-parameter"
    "-Wno-unused-variable"
    "-Wno-undef"
    "-fvisibility=hidden"
    # NOTE: The RTTI setting must match what LLVM was compiled with (defaults
    # to RTTI disabled).
    "$<$<COMPILE_LANGUAGE:CXX>:-fno-rtti>"
    "$<$<COMPILE_LANGUAGE:CXX>:-fno-exceptions>"

  MSVC_OR_CLANG_CL
    # Default warning level (severe + significant + production quality).
    # This does not include level 4, "informational", warnings or those that
    # are off by default.
    # https://docs.microsoft.com/en-us/cpp/build/reference/compiler-option-warning-level
    # Note that we set CMake policy CMP0092 (if found), making this explicit:
    # https://cmake.org/cmake/help/v3.15/policy/CMP0092.html
    "/W3"

    # Exclude a bunch of rarely-used APIs, such as crypto/DDE/shell.
    # https://docs.microsoft.com/en-us/windows/win32/winprog/using-the-windows-headers
    # NOTE: this is not really required anymore for build performance but does
    # work around some issues that crop up with header version compatibility
    # (abseil has issues with winsock versions).
    "/DWIN32_LEAN_AND_MEAN"

    # Don't allow windows.h to define MIN and MAX and conflict with the STL.
    # There's no legit use for these macros as any code we are writing ourselves
    # that we want a MIN/MAX in should be using an IREE-prefixed version
    # instead: iree_min iree_max
    # https://stackoverflow.com/a/4914108
    "/DNOMINMAX"

    # Adds M_PI and other constants to <math.h>/<cmath> (to match non-windows).
    # https://docs.microsoft.com/en-us/cpp/c-runtime-library/math-constants
    "/D_USE_MATH_DEFINES"

    # Disable the "deprecation" warnings about CRT functions like strcpy.
    # Though the secure versions *are* better, they aren't portable and as such
    # just make cross-platform code annoying. One solution is to reimplement
    # them in a portable fashion and use those - and that's what we try to do
    # in certain places where we can get away with it. Other uses, like getenv,
    # are fine as these are not intended for use in core runtime code that needs
    # to be secure (friends don't let friends ship entire compiler stacks
    # embedded inside security sensitive applications anyway :).
    # https://docs.microsoft.com/en-us/cpp/c-runtime-library/security-features-in-the-crt
    "/D_CRT_SECURE_NO_WARNINGS"

    # With the above said about the "deprecated" functions; this useful flag
    # will at least try to use them when possible without any change to user
    # code. Note however because the new versions use templates they won't be
    # activated in C code; that's fine.
    # https://docs.microsoft.com/en-us/cpp/c-runtime-library/secure-template-overloads
    "/D_CRT_SECURE_CPP_OVERLOAD_STANDARD_NAMES"

    # Configure RTTI generation.
    # - /GR - Enable generation of RTTI (default)
    # - /GR- - Disables generation of RTTI
    # https://docs.microsoft.com/en-us/cpp/build/reference/gr-enable-run-time-type-information?view=msvc-160
    "/GR-"

    # Default max section count is 64k, which is woefully inadequate for some of
    # the insanely bloated tablegen outputs LLVM/MLIR produces. This cranks it
    # up to 2^32. It's not great that we have to generate/link files like that
    # but it's better to not get spurious failures during LTCG.
    # https://docs.microsoft.com/en-us/cpp/build/reference/bigobj-increase-number-of-sections-in-dot-obj-file
    "/bigobj"

    # "nonstandard extension used : zero-sized array in struct/union"
    # This happens with unsized or zero-length arrays at the end of structs,
    # which is completely valid in C where we do it and get this warning. Shut
    # it up and rely on the better warnings from clang to catch if we try to
    # use it where it really matters (on a class that has copy/move ctors, etc).
    # https://docs.microsoft.com/en-us/cpp/error-messages/compiler-warnings/compiler-warning-levels-2-and-4-c4200
    "/wd4200"

    # "signed/unsigned mismatch in comparison"
    # This is along the lines of a generic implicit conversion warning but tends
    # to crop up in code that implicitly treats unsigned size_t values as if
    # they were signed values instead of properly using ssize_t. In certain
    # cases where the comparison being performed may be guarding access to
    # memory this can cause unexpected behavior ("-1ull < 512ull, great let's
    # dereference buffer[-1ull]!").
    # https://docs.microsoft.com/en-us/cpp/error-messages/compiler-warnings/compiler-warning-level-3-c4018
    #
    # TODO(#3844): remove this (or make it per-file to iree/compiler, as LLVM
    # tends to not care about these kind of things and it crops up there a lot).
    "/wd4018"

    # Also common in LLVM is mismatching signed/unsigned math. That's even more
    # dangerous than C4018: almost always these crop up in doing something with
    # a size_t and a non-size_t value (usually int or something like it) and do
    # you want out-of-bounds access exploits? Because that's how you get
    # out-of-bounds access exploits. Before fuzzers took over finding code and
    # trying to compile it with this warning forced to be an error was a way to
    # narrow down the places to look for attack vectors. I lived through the
    # Microsoft SAL/safe-int code red, and once you get used to using the safe
    # buffer offset/size manipulation functions it eliminates all kinds of
    # annoying bugs - as well as potential security issues.
    #
    # TODO(#3844): work to remove this class of errors from our code. It's
    # almost entirely in LLVM related stuff so per-file iree/compiler/... would
    # be fine.
    "/wd4146"  # operator applied to unsigned type, result still unsigned
    "/wd4244"  # possible loss of data
    "/wd4267"  # initializing: possible loss of data

    # Misc tweaks to better match reasonable clang/gcc behavior:
    "/wd4005"  # allow: macro redefinition
    "/wd4065"  # allow: switch statement contains 'default' but no 'case' labels
    "/wd4141"  # allow: inline used more than once
    "/wd4624"  # allow: destructor was implicitly defined as deleted

    # TODO(benvanik): confirm these are all still required and document:
    "/wd4146"  # operator applied to unsigned type, result still unsigned
    "/wd4244"  # possible loss of data
    "/wd4267"  # initializing: possible loss of data
    "/wd5105"  # allow: macro expansion producing 'defined' has undefined behavior
)

# On MSVC, CMake sets /GR by default (enabling RTTI), but we set /GR-
# (disabling it) above. To avoid Command line warning D9025 which warns about
# overriding the flag value, we remove /GR from global CMake flags.
#
# Note: this may have ripple effects on downstream projects using IREE. If this
# is a problem for your project, please reach out to us and we'll figure out a
# compatible solution.
#
# See also:
#   https://github.com/google/iree/issues/4665.
#   https://discourse.cmake.org/t/how-to-fix-build-warning-d9025-overriding-gr-with-gr/878
#   https://gitlab.kitware.com/cmake/cmake/-/issues/20610
if(CMAKE_CXX_FLAGS AND "${CMAKE_CXX_COMPILER_ID}" STREQUAL "MSVC")
  string(REPLACE "/GR" "" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
endif()

if(NOT ANDROID)
  iree_select_compiler_opts(_IREE_PTHREADS_LINKOPTS
    CLANG_OR_GCC
      "-lpthread"
  )
else()
  # Android provides its own pthreads support with no linking required.
endif()

if(ANDROID)
  # logging.h on Android needs llog to link in Android logging.
  iree_select_compiler_opts(_IREE_LOGGING_LINKOPTS
    CLANG_OR_GCC
      "-llog"
  )
endif()

iree_select_compiler_opts(IREE_DEFAULT_LINKOPTS
  ALL
    # TODO(benvanik): remove the ABSL usage here; we aren't abseil.
    "${ABSL_DEFAULT_LINKOPTS}"
  CLANG_OR_GCC
    # Required by all modern software, effectively:
    "-ldl"
    "-lm"
    ${_IREE_PTHREADS_LINKOPTS}
    ${_IREE_LOGGING_LINKOPTS}
)

# Add to LINKOPTS on a binary to configure it for X/Wayland/Windows/etc
# depending on the target cross-compilation platform.
if(${CMAKE_SYSTEM_NAME} STREQUAL "Windows")
  set(IREE_TARGET_GUI_LINKOPTS "-SUBSYSTEM:WINDOWS")
else()
  set(IREE_TARGET_GUI_LINKOPTS "")
endif()

# TODO(benvanik): remove the ABSL usage here; we aren't abseil.
set(IREE_TEST_COPTS "${ABSL_TEST_COPTS}")

#-------------------------------------------------------------------------------
# Size-optimized build flags
#-------------------------------------------------------------------------------

  # TODO(#898): add a dedicated size-constrained configuration.
if(${IREE_SIZE_OPTIMIZED})
  iree_select_compiler_opts(IREE_SIZE_OPTIMIZED_DEFAULT_COPTS
    MSVC_OR_CLANG_CL
      "/GS-"
      "/GL"
      "/Gw"
      "/Gy"
      "/DNDEBUG"
      "/DIREE_STATUS_MODE=0"
      "/DIREE_FLAGS_ENABLE_CLI=0"
      "/DIREE_HAL_MODULE_STRING_UTIL_ENABLE=0"
      "/DIREE_VM_EXT_I64_ENABLE=0"
      "/DIREE_VM_EXT_F32_ENABLE=0"
      "/Os"
      "/Oy"
      "/Zi"
      "/c"
  )
  iree_select_compiler_opts(IREE_SIZE_OPTIMIZED_DEFAULT_LINKOPTS
    MSVC_OR_CLANG_CL
      "-DEBUG:FULL"
      "-LTCG"
      "-opt:ref,icf"
  )
  # TODO(#898): make this only impact the runtime (IREE_RUNTIME_DEFAULT_...).
  set(IREE_DEFAULT_COPTS
      "${IREE_DEFAULT_COPTS}"
      "${IREE_SIZE_OPTIMIZED_DEFAULT_COPTS}")
  set(IREE_DEFAULT_LINKOPTS
      "${IREE_DEFAULT_LINKOPTS}"
      "${IREE_SIZE_OPTIMIZED_DEFAULT_LINKOPTS}")
endif()

#-------------------------------------------------------------------------------
# Compiler: Clang/LLVM
#-------------------------------------------------------------------------------

# TODO(benvanik): Clang/LLVM options.

#-------------------------------------------------------------------------------
# Compiler: GCC
#-------------------------------------------------------------------------------

# TODO(benvanik): GCC options.

#-------------------------------------------------------------------------------
# Compiler: MSVC
#-------------------------------------------------------------------------------

# TODO(benvanik): MSVC options.

#-------------------------------------------------------------------------------
# Third party: benchmark
#-------------------------------------------------------------------------------

set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "" FORCE)
set(BENCHMARK_ENABLE_INSTALL OFF CACHE BOOL "" FORCE)

#-------------------------------------------------------------------------------
# Third party: cpuinfo
#-------------------------------------------------------------------------------

set(CPUINFO_BUILD_TOOLS ON CACHE BOOL "" FORCE)

set(CPUINFO_BUILD_BENCHMARKS OFF CACHE BOOL "" FORCE)
set(CPUINFO_BUILD_UNIT_TESTS OFF CACHE BOOL "" FORCE)
set(CPUINFO_BUILD_MOCK_TESTS OFF CACHE BOOL "" FORCE)

#-------------------------------------------------------------------------------
# Third party: flatcc
#-------------------------------------------------------------------------------

set(FLATCC_TEST OFF CACHE BOOL "" FORCE)
set(FLATCC_CXX_TEST OFF CACHE BOOL "" FORCE)
set(FLATCC_REFLECTION OFF CACHE BOOL "" FORCE)
set(FLATCC_ALLOW_WERROR OFF CACHE BOOL "" FORCE)

if(CMAKE_CROSSCOMPILING)
  set(FLATCC_RTONLY ON CACHE BOOL "" FORCE)
else()
  set(FLATCC_RTONLY OFF CACHE BOOL "" FORCE)
endif()

#-------------------------------------------------------------------------------
# Third party: gtest
#-------------------------------------------------------------------------------

set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)

#-------------------------------------------------------------------------------
# Third party: llvm/mlir
#-------------------------------------------------------------------------------

set(LLVM_INCLUDE_EXAMPLES OFF CACHE BOOL "" FORCE)
set(LLVM_INCLUDE_TESTS OFF CACHE BOOL "" FORCE)
set(LLVM_INCLUDE_BENCHMARKS OFF CACHE BOOL "" FORCE)
set(LLVM_APPEND_VC_REV OFF CACHE BOOL "" FORCE)
set(LLVM_ENABLE_IDE ON CACHE BOOL "" FORCE)

# TODO(ataei): Use optional build time targets selection for LLVMAOT.
set(LLVM_TARGETS_TO_BUILD "WebAssembly;X86;ARM;AArch64;RISCV;NVPTX"
    CACHE STRING "" FORCE)

set(LLVM_ENABLE_PROJECTS "mlir" CACHE STRING "" FORCE)
set(LLVM_ENABLE_BINDINGS OFF CACHE BOOL "" FORCE)

if(IREE_USE_LINKER)
  set(LLVM_USE_LINKER ${IREE_USE_LINKER} CACHE STRING "" FORCE)
endif()

set(MLIR_TABLEGEN_EXE mlir-tblgen)
# iree-tblgen is not defined using the add_tablegen mechanism as other TableGen
# tools in LLVM.
iree_get_executable_path(IREE_TABLEGEN_EXE iree-tblgen)

#-------------------------------------------------------------------------------
# Third party: mlir-emitc
#-------------------------------------------------------------------------------

if(IREE_ENABLE_EMITC)
  set(EMITC_BUILD_EMBEDDED ON)
  set(EMITC_ENABLE_HLO OFF)
  set(EMITC_INCLUDE_TESTS OFF)

  add_definitions(-DIREE_HAVE_EMITC_DIALECT)
endif()

#-------------------------------------------------------------------------------
# Third party: SPIRV-Cross
#-------------------------------------------------------------------------------

if(${IREE_TARGET_BACKEND_METAL-SPIRV})
  set(SPIRV_CROSS_ENABLE_MSL ON CACHE BOOL "" FORCE)
  set(SPIRV_CROSS_ENABLE_GLSL ON CACHE BOOL "" FORCE) # Required to enable MSL

  set(SPIRV_CROSS_EXCEPTIONS_TO_ASSERTIONS OFF CACHE BOOL "" FORCE)
  set(SPIRV_CROSS_CLI OFF CACHE BOOL "" FORCE)
  set(SPIRV_CROSS_ENABLE_TESTS OFF CACHE BOOL "" FORCE)
  set(SPIRV_CROSS_SKIP_INSTALL ON CACHE BOOL "" FORCE)

  set(SPIRV_CROSS_ENABLE_HLSL OFF CACHE BOOL "" FORCE)
  set(SPIRV_CROSS_ENABLE_CPP OFF CACHE BOOL "" FORCE)
  set(SPIRV_CROSS_ENABLE_REFLECT OFF CACHE BOOL "" FORCE)
  set(SPIRV_CROSS_ENABLE_C_API OFF CACHE BOOL "" FORCE)
  set(SPIRV_CROSS_ENABLE_UTIL OFF CACHE BOOL "" FORCE)
endif()

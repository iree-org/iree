// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_UTILS_CUSTOMKERNELTARGETINFO_H_
#define IREE_COMPILER_UTILS_CUSTOMKERNELTARGETINFO_H_

#include <stdint.h>

#include <cassert>

#include "mlir/Support/LogicalResult.h"

namespace mlir {
namespace iree_compiler {

// Enumerates target ISAs that we care about. 'int8_t' because we somewhat
// care because this is used in struct MMTKernel, which is passed by value.
enum class CustomKernelTargetArch : int8_t { None, Aarch64 };

// Enumerates arch-specific target features that we care about.
// We explicitly want to stick to the default enumeration values (0, 1, 2, ...,
// no greater than 63) because this is going to be indexing a uint64 bitfield.
// Intentionally not reusing bits across architectures to be able to catch
// most bugs. 64 is enough across all target architectures for now.
enum class CustomKernelTargetFeature {
  // Indicates a preference for intrinsics over inline asm. Unlike other bits,
  // this is generic, not tied to a particular architecture or CPU feature, and
  // it has to be passed through some custom boolean flag or option, not as
  // part of the target CPU features.
  Intrinsics,
  // Aarch64 features.
  Aarch64Dotprod,
};

inline bool isFeatureForArch(CustomKernelTargetFeature feature,
                             CustomKernelTargetArch arch) {
  switch (feature) {
    case CustomKernelTargetFeature::Intrinsics:
      return true;
    case CustomKernelTargetFeature::Aarch64Dotprod:
      return arch == CustomKernelTargetArch::Aarch64;
  }
  assert(false && "Unhandled CustomKernelTargetFeature value");
  return false;
}

// Class used to pass some target information to patterns/passes that need it.
// The information could come from pass options, e.g.
//    -iree-llvmcpu-vector-contract-custom-kernels='arch=aarch64
//    features=+dotprod intrinsics'
// or from a parent HAL::ExecutableVariantOp and/or be complemented by a
// global flag like clMmt4dUseIntrinsics.
class CustomKernelsTargetInfo {
 public:
  void init(CustomKernelTargetArch a) {
    assert(arch == CustomKernelTargetArch::None);
    arch = a;
  }
  bool is(CustomKernelTargetArch a) const { return arch == a; }
  bool has(CustomKernelTargetFeature f) const {
    if (!isFeatureForArch(f, arch)) {
      return false;
    }
    return features & (1ull << static_cast<int>(f));
  }
  void add(CustomKernelTargetFeature f) {
    assert(isFeatureForArch(f, arch));
    features |= (1ull << static_cast<int>(f));
  }

 private:
  CustomKernelTargetArch arch = CustomKernelTargetArch::None;
  // Bitfield, with bits indexed by CustomKernelTargetFeature.
  uint64_t features = 0;
};

LogicalResult ParseCustomKernelsTargetInfo(llvm::StringRef archStr,
                                           llvm::StringRef featuresStr,
                                           CustomKernelsTargetInfo &targetInfo);

}  // namespace iree_compiler
}  // namespace mlir

#endif  // IREE_COMPILER_UTILS_CUSTOMKERNELTARGETINFO_H_

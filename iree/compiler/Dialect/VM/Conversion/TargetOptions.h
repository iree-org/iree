// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_DIALECT_VM_CONVERSION_TARGETOPTIONS_H_
#define IREE_COMPILER_DIALECT_VM_CONVERSION_TARGETOPTIONS_H_

#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace VM {

// Defines runtime VM extension opcode sets.
enum class OpcodeExtension {
  // Adds ops for manipulating i64 types.
  kI64,
  // Adds ops for manipulating f32 types.
  kF32,
  // Adds ops for manipulating f64 types.
  kF64,
};

// Controls VM translation targets.
struct TargetOptions {
  // Target size of `index` when converted to an integer in bits.
  int indexBits = 32;

  // Whether the i64 extension is enabled in the target VM.
  bool i64Extension = false;
  // Whether the f32 extension is enabled in the target VM.
  bool f32Extension = false;
  // Whether the f64 extension is enabled in the target VM.
  bool f64Extension = false;

  // Whether to truncate i64 types to i32 when the i64 extension is not
  // enabled.
  bool truncateUnsupportedIntegers = true;
  // Whether to truncate f64 types to f32 when the f64 extension is not
  // enabled.
  bool truncateUnsupportedFloats = true;

  // Prefer optimizations that reduce VM stack usage over performance.
  bool optimizeForStackSize = true;
};

// Returns a TargetOptions struct initialized with the
// --iree-vm-target-* flags.
TargetOptions getTargetOptionsFromFlags();

}  // namespace VM
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir

#endif  // IREE_COMPILER_DIALECT_VM_CONVERSION_TARGETOPTIONS_H_

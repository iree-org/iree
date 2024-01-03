// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_DIALECT_VM_CONVERSION_TARGETOPTIONS_H_
#define IREE_COMPILER_DIALECT_VM_CONVERSION_TARGETOPTIONS_H_

#include "iree/compiler/Utils/OptionUtils.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir::iree_compiler::IREE::VM {

// Controls VM translation targets.
struct TargetOptions {
  // Target size of `index` when converted to an integer in bits.
  int indexBits = 64;

  // Whether the f32 extension is enabled in the target VM.
  bool f32Extension = true;
  // Whether the f64 extension is enabled in the target VM.
  bool f64Extension = false;

  // Whether to truncate f64 types to f32 when the f64 extension is not
  // enabled.
  bool truncateUnsupportedFloats = true;

  // Prefer optimizations that reduce VM stack usage over performance.
  bool optimizeForStackSize = false;

  void bindOptions(OptionsBinder &binder);
  using FromFlags = OptionsFromFlags<TargetOptions>;
};

} // namespace mlir::iree_compiler::IREE::VM

#endif // IREE_COMPILER_DIALECT_VM_CONVERSION_TARGETOPTIONS_H_

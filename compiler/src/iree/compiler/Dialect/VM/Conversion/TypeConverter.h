// Copyright 2019 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_DIALECT_VM_CONVERSION_TYPECONVERTER_H_
#define IREE_COMPILER_DIALECT_VM_CONVERSION_TYPECONVERTER_H_

#include "iree/compiler/Dialect/VM/Conversion/TargetOptions.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir::iree_compiler::IREE::VM {

class TypeConverter : public mlir::TypeConverter {
public:
  explicit TypeConverter(TargetOptions targetOptions);

  const TargetOptions &targetOptions() const { return targetOptions_; }

private:
  TargetOptions targetOptions_;
};

// Returns whether an arithmetic op producing `sourceType` can lower to VM
// arithmetic on operands of the converted type `convertedType`. VM arithmetic
// runs on integers and floats, and a float op must not run on the storage bits
// of a narrow float that the VM holds in an integer register.
bool isVMArithmeticType(Type sourceType, Type convertedType);

} // namespace mlir::iree_compiler::IREE::VM

#endif // IREE_COMPILER_DIALECT_VM_CONVERSION_TYPECONVERTER_H_

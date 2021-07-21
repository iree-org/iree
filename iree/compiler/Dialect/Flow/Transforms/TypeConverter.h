// Copyright 2019 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_DIALECT_FLOW_CONVERSION_TYPECONVERTER_H_
#define IREE_COMPILER_DIALECT_FLOW_CONVERSION_TYPECONVERTER_H_

#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
namespace iree_compiler {

class FlowTypeConverter : public TypeConverter {
 public:
  FlowTypeConverter();
};

}  // namespace iree_compiler
}  // namespace mlir

#endif  // IREE_COMPILER_DIALECT_FLOW_CONVERSION_TYPECONVERTER_H_

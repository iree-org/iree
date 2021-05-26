// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_DIALECT_SHAPE_IR_XLAHLOSHAPEBUILDER_H_
#define IREE_COMPILER_DIALECT_SHAPE_IR_XLAHLOSHAPEBUILDER_H_

#include "iree/compiler/Dialect/Shape/IR/ShapeInterface.h"

namespace mlir {
namespace mhlo {

// Creates a custom op shape builder for XLA-HLO ops that are not otherwise
// supported through traits or other declarative means.
void populateXlaHloCustomOpShapeBuilder(
    iree_compiler::Shape::CustomOpShapeBuilderList &builders);

}  // namespace mhlo
}  // namespace mlir

#endif  // IREE_COMPILER_DIALECT_SHAPE_IR_XLAHLOSHAPEBUILDER_H_

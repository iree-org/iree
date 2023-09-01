//===-------------------------------------------------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Also available under a BSD-style license. See LICENSE.
//
//===----------------------------------------------------------------------===//

#ifndef TORCH_MLIR_DIALECT_TMTENSOR_BASE
#define TORCH_MLIR_DIALECT_TMTENSOR_BASE

include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// Dialect definition
//===----------------------------------------------------------------------===//

def TMTensor_Dialect : Dialect {
  let name = "tm_tensor";
  let cppNamespace = "::mlir::torch::TMTensor";
  let description = [{
    The tm_tensor (tm = torch-mlir) dialect is a temporary staging ground in
    the torch-mlir project for a set of widely-accepted tensor compute
    operations that are not well-served by existing representations in MLIR
    upstream. These ops are currently heavily inspired by the linalg_ext
    dialect (which itself is heavily inspired by the structured ops of the
    linalg dialect). But while linalg_ext is meant to power specific codegen
    transformations, the tm_tensor dialect is a much more pure "interface
    dialect" agnostic to any particular set of transformations applied to
    the operations. We simply require a way to name the specified operations
    for interchange between projects, without taking strong opinions on the
    mechanics of transformations.

    The dialect does include interfaces to generate scalar reference code for
    the operations, which simultaneously provides a precise definition of their
    semantics, and aids in producing executable reference implementations of
    the operations.

    The goal of this dialect is to eventually either be upstreamed or to be
    subsumed by functionality included by upstream MLIR. It should also be kept
    consistent with the linalg_ext dialect unless there is a good reason not
    to.
  }];
  let hasCanonicalizer = 1;
}

//===----------------------------------------------------------------------===//
// Type definitions
//===----------------------------------------------------------------------===//

class RankedTensorOrMemRefOf<list<Type> allowedTypes> :
  ShapedContainerType<allowedTypes,
      Or<[IsMemRefTypePred, And<[IsTensorTypePred, HasRankPred]>]>,
  "ranked tensor or memref", "::mlir::ShapedType">;

def AnyRankedTensorOrMemRefType : RankedTensorOrMemRefOf<[AnyType]>;

#endif // TORCH_MLIR_DIALECT_TMTENSOR_BASE

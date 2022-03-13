//===- TransformOpInterface.td - Interfaces for xform ops --*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Defines interfaces and traits for operations representing controllable
// transformations.
//
//===----------------------------------------------------------------------===//

#ifndef TRANSFORM_OP_INTERFACE
#define TRANSFORM_OP_INTERFACE

include "mlir/IR/OpBase.td"

def TransformOpInterface : OpInterface<"TransformOpInterface"> {
  let description = [{This interface is to be implemented by operations that
  identify transformations to be performed on other operations. The former
  are referred to as transform IR operations. The latter are referred to as
  payload IR operations. Such transform IR operations provide a fine-grain
  control mechanism over how transformations are applied by using and defining
  transform IR values, referred to as handles, that correspond to sets of
  operations in the payload IR. Transformations are applied starting from
  the operations identified by handles, but may affect other operations as
  well. Further restrictions may be imposed by flows that rely on transform
  IR operations to control transformations.
  }];

  let cppNamespace = "::mlir::linalg::transform";

  let methods = [
    InterfaceMethod<
      /*desc=*/[{Applies the transformation represented by the current
      operation. This accepts as arguments the object that must be populated
      with results of the current transformation and a transformation state
      object that can be used for queries, e.g., to obtain the list of
      operations on which the transformation represented by the current op is
      targeted.}],
      /*returnType=*/"LogicalResult",
      /*name=*/"apply",
      /*arguments=*/(ins
          "::mlir::linalg::transform::TransformResults &":$transformResults,
          "::mlir::linalg::transform::TransformState &":$state)
    >,
  ];
}

def TargetableSingleOperandTransformOpTrait
    : NativeOpTrait<"TargetableSingleOperandOpTrait"> {
  let cppNamespace = "::mlir::linalg::transform";
}

#endif // TRANSFORM_OP_INTERFACE

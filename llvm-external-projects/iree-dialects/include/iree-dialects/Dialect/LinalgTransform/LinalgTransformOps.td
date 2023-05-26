// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LINALG_TRANSFORM_OPS
#define LINALG_TRANSFORM_OPS

include "mlir/IR/OpBase.td"
include "mlir/IR/OpAsmInterface.td"
include "mlir/Dialect/PDL/IR/PDLTypes.td"
include "mlir/Dialect/Transform/IR/TransformInterfaces.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"

def Linalg_Transform_Dialect : Dialect {
  let name = "iree_linalg_transform";
  let cppNamespace = "::mlir::linalg::transform";
  let dependentDialects = [
    "linalg::LinalgDialect",
  ];
}

// Operations with this trait must provide the following methods:
//   - `Value target()` - returns the operation handle (value of !pdl.operation
//     type) targeted by this transformation, if available;
//   - `Optional<SymbolRefAttr> matcher()` - returns the name of the PDL matcher
//     that selects the ops targeted by this transformation, if provided.
class Linalg_Transform_Operation<string name, list<Trait> props = []>
    : Op<Linalg_Transform_Dialect, name, props> {
  let cppNamespace = "::mlir::linalg::transform";
}

class Transform_Op<string name, list<Trait> props = []>
    : Linalg_Transform_Operation<name, !listconcat(props, [
      DeclareOpInterfaceMethods<TransformOpInterface, ["apply"]>])>;

//===----------------------------------------------------------------------===//

def ScopeOp : Linalg_Transform_Operation<"util.scope",
    [IsolatedFromAbove, DeclareOpInterfaceMethods<RegionBranchOpInterface>]> {
  let description = [{An operation to restrict transformation scopes.}];

  let regions = (region AnyRegion:$body);
  let arguments = (ins Variadic<AnyType>:$ins);
  let results = (outs Variadic<AnyType>:$outs);
  let assemblyFormat = [{ `(` operands `)` attr-dict-with-keyword $body
                          `:` functional-type(operands, results) }];
}

def ForwardOp : Linalg_Transform_Operation<"util.forward",
                                           [Terminator, HasParent<"ScopeOp">]> {
  let description = [{Terminator for a scope operation, indicating the results
                      that should be forwarded out of the scope.}];

  let arguments = (ins Variadic<AnyType>:$ins);
  let assemblyFormat = "operands attr-dict `:` type(operands)";
}

//===----------------------------------------------------------------------===//

def ExpertOp : Linalg_Transform_Operation<"expert"> {
  let description = [{A "transformation expert" that can be lowered to a
  sequence of transformations. The details of the lowering depend on the name
  and are expressed declaratively.}];

  let arguments = (ins TransformHandleTypeInterface:$target,
                   StrAttr:$expertName);
  let results = (outs TransformHandleTypeInterface:$transformed);

  let assemblyFormat = "`apply` $expertName `to` $target attr-dict `:` functional-type(operands, results)";
}

#endif // LINALG_TRANSFORM_OPS

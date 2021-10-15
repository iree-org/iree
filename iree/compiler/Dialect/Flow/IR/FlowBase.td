// Copyright 2019 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_DIALECT_FLOW_BASE
#define IREE_DIALECT_FLOW_BASE

include "iree/compiler/Dialect/Flow/IR/FlowInterfaces.td"
include "iree/compiler/Dialect/Util/IR/UtilBase.td"
include "iree/compiler/Dialect/Shape/IR/ShapeBase.td"

//===----------------------------------------------------------------------===//
// IREE execution flow dialect
//===----------------------------------------------------------------------===//

def FLOW_Dialect : Dialect {
  let name = "flow";
  let cppNamespace = "::mlir::iree_compiler::IREE::Flow";

  let summary = [{
    A dialect designed to model execution data flow and partitioning.
  }];
  let description = [{
    The flow dialect is used to model regions of dense computation and the data
    flow between them. MLIR value-semantic tensors are used as the primary data
    type to allow SSA use-def to provide a bulk of the infrastructure required
    to perform the computation partitioning and outlining.

    The dialect is designed to ingest relatively high-level linear algebra via
    XLA HLO ops (that also operate on the value-semantic tensor types) and
    optionally MLIR standard ops for control flow and other actions. After
    conversion of any higher-level ops that have special semantics in the flow
    dialect, such as global variables, the rest are partitioned into regions
    containing simple and compatible computations. Finally, outlining moves the
    computations into executables and leaves only the execution flow encoded via
    dispatch operations.

    The primary unit of interest is a "dispatch region" containing compatible
    computations that can be scheduled together efficiently (and safely).
    "Compatible" here is specified as similarly shaped workloads that indicate
    how many invocations a computation can be parallelized across when running
    in a SPMD execution model. Though it depends on the particular runtime
    backends this more concretely means things like the untiled workload
    (or tiled workgroups) used in GPU dispatches or similar thread pool
    executors.

    After identification of the dispatchable regions a set of transformations
    performs folding and simplification to reduce the total number of
    dispatches. Heuristics are used in certain cases to more efficiently
    schedule special ops (such as GEMM) and the design is amenable to profile-
    guided analysis that can be added in the future.

    The resulting outlined executable modules containing the dispatchable code
    can be translated to one or more backends (such as SPIR-V for Vulkan, or
    LLVM IR for running on the CPU, etc). The IR that is outlined is untouched
    and in the input format (such as XLA HLO ops) allowing conversion using any
    MLIR target that supports ingesting such input. A few special ops are used
    to communicate statically available information such as the expected
    workload size, shapes of inputs and outputs, etc.
  }];
}

//===----------------------------------------------------------------------===//
// Base flow dialect op classes
//===----------------------------------------------------------------------===//

class FLOW_Op<string mnemonic, list<OpTrait> traits = []> :
    Op<FLOW_Dialect, mnemonic, traits> {
  let parser = [{ return parse$cppClass(parser, &result); }];
  let printer = [{ return print$cppClass(p, *this); }];
}

//===----------------------------------------------------------------------===//
// Flow dialect types
//===----------------------------------------------------------------------===//

def FLOW_PrimitiveType : AnyTypeOf<[Index, AnySignlessInteger, AnyFloat]>;

def FLOW_Dim : TypeAlias<Index>;
def FLOW_ShapeDynamicDims : Variadic<FLOW_Dim>;

def FLOW_Tensor : TypeAlias<AnyRankedTensor>;

def FLOW_ExecutableRefAttr : AliasedSymbolRefAttr;
def FLOW_GlobalRefAttr : AliasedSymbolRefAttr;
def FLOW_GlobalPtr : Util_AnyPtrOf<[FLOW_Tensor, FLOW_PrimitiveType]>;

//===----------------------------------------------------------------------===//
// Dispatch types
//===----------------------------------------------------------------------===//

def FLOW_TensorAccessCanReadPred : CPred<[{
  $_self.cast<IREE::Flow::DispatchTensorType>().getAccess() ==
      IREE::Flow::TensorAccess::ReadOnly ||
  $_self.cast<IREE::Flow::DispatchTensorType>().getAccess() ==
      IREE::Flow::TensorAccess::ReadWrite
}]>;

def FLOW_TensorAccessCanWritePred : CPred<[{
  $_self.cast<IREE::Flow::DispatchTensorType>().getAccess() ==
      IREE::Flow::TensorAccess::WriteOnly ||
  $_self.cast<IREE::Flow::DispatchTensorType>().getAccess() ==
      IREE::Flow::TensorAccess::ReadWrite
}]>;

def FLOW_DispatchTensor : DialectType<
    FLOW_Dialect,
    CPred<"$_self.isa<IREE::Flow::DispatchTensorType>()">,
    "dispatch.tensor"> {
  let description = [{
    A placeholder for a dispatch region input/output operand. This can be used
    to query the metadata about the tensor (such as its shape) as well as both
    load and store from the backing tensor representation.
  }];
}

def FLOW_ReadableDispatchTensor : DialectType<
    FLOW_Dialect,
    And<[
      CPred<"$_self.isa<IREE::Flow::DispatchTensorType>()">,
      FLOW_TensorAccessCanReadPred,
    ]>,
    "dispatch.tensor"> {
  let description = [{
    A placeholder for a dispatch region input operand. This can be used
    to query the metadata about the tensor (such as its shape) as well as load
    from the backing tensor representation.
  }];
}

def FLOW_WritableDispatchTensor : DialectType<
    FLOW_Dialect,
    And<[
      CPred<"$_self.isa<IREE::Flow::DispatchTensorType>()">,
      FLOW_TensorAccessCanWritePred,
    ]>,
    "dispatch.tensor"> {
  let description = [{
    A placeholder for a dispatch region output operand. This can be used
    to query the metadata about the tensor (such as its shape) as well as store
    to the backing tensor representation.
  }];
}

// TODO(benvanik): remove when we have real types using this.
def FLOW_Dummy0 : TypeDef<FLOW_Dialect, "Dummy", []> {
  let mnemonic = "dummy";
}
def FLOW_Dummy1 : AttrDef<FLOW_Dialect, "Dummy", []> {
  let mnemonic = "dummy";
}

#endif  // IREE_DIALECT_FLOW_BASE

// Copyright 2019 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_DIALECT_FLOW_BASE
#define IREE_DIALECT_FLOW_BASE

include "iree/compiler/Dialect/Flow/IR/FlowInterfaces.td"
include "iree/compiler/Dialect/Util/IR/UtilBase.td"
include "iree/compiler/Dialect/Util/IR/UtilTypes.td"
include "mlir/IR/AttrTypeBase.td"

//===----------------------------------------------------------------------===//
// IREE execution flow dialect
//===----------------------------------------------------------------------===//

def Flow_Dialect : Dialect {
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

  let useDefaultTypePrinterParser = 0;
  let useDefaultAttributePrinterParser = 1;
}

//===----------------------------------------------------------------------===//
// Base flow dialect op classes
//===----------------------------------------------------------------------===//

class FLOW_Op<string mnemonic, list<Trait> traits = []> :
    Op<Flow_Dialect, mnemonic, traits> {
  let hasCustomAssemblyFormat = 1;
}

//===----------------------------------------------------------------------===//
// Flow dialect types
//===----------------------------------------------------------------------===//

def FLOW_PrimitiveType : AnyTypeOf<[Index, AnySignlessInteger, AnyFloat]>;

def FLOW_Dim : TypeAlias<Index>;
def FLOW_ShapeDynamicDims : Variadic<FLOW_Dim>;

def FLOW_Tensor : TypeAlias<AnyRankedTensor>;

def FLOW_ExecutableRefAttr : Util_AliasedSymbolRefAttr;
def FLOW_GlobalRefAttr : Util_AliasedSymbolRefAttr;
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
    Flow_Dialect,
    CPred<"$_self.isa<IREE::Flow::DispatchTensorType>()">,
    "dispatch.tensor"> {
  let description = [{
    A placeholder for a dispatch region input/output operand. This can be used
    to query the metadata about the tensor (such as its shape) as well as both
    load and store from the backing tensor representation.
  }];
}

def FLOW_ReadableDispatchTensor : DialectType<
    Flow_Dialect,
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
    Flow_Dialect,
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
def FLOW_Dummy0 : TypeDef<Flow_Dialect, "Dummy", []> {
  let mnemonic = "dummy";
}
def FLOW_Dummy1 : AttrDef<Flow_Dialect, "Dummy", []> {
  let mnemonic = "dummy";
}

//===----------------------------------------------------------------------===//
// Flow enums
//===----------------------------------------------------------------------===//

def FLOW_CollectiveElementType_Sint8 : I32EnumAttrCase<"Sint8", 0, "si8">;
def FLOW_CollectiveElementType_Uint8 : I32EnumAttrCase<"Uint8", 1, "ui8">;
def FLOW_CollectiveElementType_Sint16 : I32EnumAttrCase<"Sint16", 2, "si16">;
def FLOW_CollectiveElementType_Uint16 : I32EnumAttrCase<"Uint16", 3, "ui16">;
def FLOW_CollectiveElementType_Sint32 : I32EnumAttrCase<"Sint32", 4, "si32">;
def FLOW_CollectiveElementType_Uint32 : I32EnumAttrCase<"Uint32", 5, "ui32">;
def FLOW_CollectiveElementType_Sint64 : I32EnumAttrCase<"Sint64", 6, "si64">;
def FLOW_CollectiveElementType_Uint64 : I32EnumAttrCase<"Uint64", 7, "ui64">;
def FLOW_CollectiveElementType_Float16 : I32EnumAttrCase<"Float16", 8, "f16">;
def FLOW_CollectiveElementType_Float32 : I32EnumAttrCase<"Float32", 9, "f32">;
def FLOW_CollectiveElementType_Float64 : I32EnumAttrCase<"Float64", 10, "f64">;
def FLOW_CollectiveElementType_BFloat16 : I32EnumAttrCase<"BFloat16", 11, "bf16">;
def FLOW_CollectiveElementTypeAttr :
    I32EnumAttr<"CollectiveElementType", "valid CollectiveElementType", [
      FLOW_CollectiveElementType_Sint8,
      FLOW_CollectiveElementType_Uint8,
      FLOW_CollectiveElementType_Sint16,
      FLOW_CollectiveElementType_Uint16,
      FLOW_CollectiveElementType_Sint32,
      FLOW_CollectiveElementType_Uint32,
      FLOW_CollectiveElementType_Sint64,
      FLOW_CollectiveElementType_Uint64,
      FLOW_CollectiveElementType_Float16,
      FLOW_CollectiveElementType_Float32,
      FLOW_CollectiveElementType_Float64,
      FLOW_CollectiveElementType_BFloat16,
    ]> {
  let cppNamespace = "::mlir::iree_compiler::IREE::Flow";
}

//===----------------------------------------------------------------------===//
// Flow channel type
//===----------------------------------------------------------------------===//

def FLOW_Channel : TypeDef<Flow_Dialect, "Channel", []> {
  let mnemonic = "channel";
  let summary = [{a collecive communication channel}];
  let description = [{
    Represents a single participant in a collective clique. Multiple channels
    may exist within the same program to allow for partial operations or
    hierarchical operations.

    In programs that have already been partitioned prior to being compiled there
    will often exist only one channel and `flow.channel.default` can be used
    to reference it. In programs that model SPMD behavior internally channels
    can be created or provided by hosting applications.
  }];
}

//===----------------------------------------------------------------------===//
// Flow collective reduction op
//===----------------------------------------------------------------------===//

// NOTE: the enum values must exactly match with the corresponding enum values
// of the Stream reduction op.

def FLOW_CollectiveReductionOp_None             : I32EnumAttrCase<"None", 0, "none">;
def FLOW_CollectiveReductionOp_ReductionSum     : I32EnumAttrCase<"ReductionSum", 1, "sum">;
def FLOW_CollectiveReductionOp_ReductionProduct : I32EnumAttrCase<"ReductionProduct", 2, "product">;
def FLOW_CollectiveReductionOp_ReductionMinimum : I32EnumAttrCase<"ReductionMinimum", 3, "minimum">;
def FLOW_CollectiveReductionOp_ReductionMaximum : I32EnumAttrCase<"ReductionMaximum", 4, "maximum">;
def FLOW_CollectiveReductionOp_ReductionAverage : I32EnumAttrCase<"ReductionAverage", 5, "average">;
def FLOW_CollectiveReductionOpAttr :
    I32EnumAttr<"CollectiveReductionOp", "valid CollectiveReductionOp", [
      FLOW_CollectiveReductionOp_None,
      FLOW_CollectiveReductionOp_ReductionSum,
      FLOW_CollectiveReductionOp_ReductionProduct,
      FLOW_CollectiveReductionOp_ReductionMinimum,
      FLOW_CollectiveReductionOp_ReductionMaximum,
      FLOW_CollectiveReductionOp_ReductionAverage,
    ]> {
  let cppNamespace = "mlir::iree_compiler::IREE::Flow";
}

#endif  // IREE_DIALECT_FLOW_BASE

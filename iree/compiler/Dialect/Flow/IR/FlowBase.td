// Copyright 2019 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef IREE_DIALECT_FLOW_BASE
#define IREE_DIALECT_FLOW_BASE

#ifndef IREE_DIALECT_COMMON_BASE
include "iree/compiler/Dialect/CommonBase.td"
#endif  // IREE_DIALECT_COMMON_BASE

//===----------------------------------------------------------------------===//
// IREE execution flow dialect
//===----------------------------------------------------------------------===//

def FLOW_Dialect : Dialect {
  let name = "flow";
  let cppNamespace = "IREE::Flow";

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

class FLOW_PureOp<string mnemonic, list<OpTrait> traits = []> :
    FLOW_Op<mnemonic, !listconcat(traits, [NoSideEffect])>;

//===----------------------------------------------------------------------===//
// Flow dialect types
//===----------------------------------------------------------------------===//

def FLOW_ExecutableRefAttr : AliasedSymbolRefAttr;
def FLOW_VariableRefAttr : AliasedSymbolRefAttr;

// TODO(benvanik): use index types instead of i32.
def FLOW_Workload : VectorOfLengthAndType<[3], [I32]> {
  let typeDescription = [{
    Describes the total untiled invocations along one or more dimensions.
    Tiling may later divide this value for workgroup sizes.
  }];
}

// TODO(cl/277443143): moving into OpBase.td.
// A `width`-bit integer elements attribute. The attribute should be
// ranked and has a shape as specified in `dims`.
class FLOW_RankedIntVectorAttr<int width, list<int> dims> : ElementsAttrBase<
  CPred<"$_self.isa<DenseIntElementsAttr>() &&"
      "$_self.cast<DenseIntElementsAttr>().getType()."
      "getElementType().isInteger(" # width # ") && "
      // Check that this is ranked and has the specified shape.
      "$_self.cast<DenseIntElementsAttr>().getType().hasRank() && "
      "$_self.cast<DenseIntElementsAttr>().getType().getShape() == "
      "llvm::ArrayRef<int64_t>({" # StrJoinInt<dims>.result # "})">,
  width # "-bit integer elements attribute of shape [" #
  StrJoinInt<dims>.result # "]"> {

  let storageType = [{ DenseIntElementsAttr }];
  let returnType = [{ DenseIntElementsAttr }];

  let constBuilderCall = "DenseElementsAttr::get("
    "VectorType::get({" # StrJoinInt<dims>.result #
    "}, $_builder.getIntegerType(" # width # ")), "
    "llvm::makeArrayRef($0)).cast<DenseIntElementsAttr>()";
  let convertFromStorage = "$_self";
}

def FLOW_WorkloadAttr : FLOW_RankedIntVectorAttr<32, [3]>;

// Use no padding and clamp the window to the valid area, possibly stopping
// early prior to having covered all data.
def FLOW_PM_ClampWindowToFit : I32EnumAttrCase<"ClampWindowToFit", 0>;
// Use initial values for padding when windows cross dimension boundaries.
def FLOW_PM_PadBorder : I32EnumAttrCase<"PadBorder", 1>;
// Describes the padding applied for a windowed operation like convolution,
// where a window is placed inside a base area.
def FLOW_PaddingModeAttr :
    I32EnumAttr<"PaddingMode", "Padding mode", [
      FLOW_PM_ClampWindowToFit,
      FLOW_PM_PadBorder,
    ]> {
  let returnType = "::mlir::iree_compiler::IREE::Flow::PaddingMode";
  let convertFromStorage = "static_cast<::mlir::iree_compiler::IREE::Flow::PaddingMode>($_self.getInt())";
  let cppNamespace = "::mlir::iree_compiler::IREE::Flow";
}

#endif  // IREE_DIALECT_FLOW_BASE

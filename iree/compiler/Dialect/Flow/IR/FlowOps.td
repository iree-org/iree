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

#ifndef IREE_DIALECT_FLOW_OPS
#define IREE_DIALECT_FLOW_OPS

#ifndef IREE_DIALECT_FLOW_BASE
include "iree/compiler/Dialect/Flow/IR/FlowBase.td"
#endif  // IREE_DIALECT_FLOW_BASE

//===----------------------------------------------------------------------===//
// Variables
//===----------------------------------------------------------------------===//

def FLOW_VariableOp : FLOW_Op<"variable", [
    Symbol,
  ]> {
  let summary = [{stateful variable declaration}];
  let description = [{
    Declares a persistent variable that maintains its value.
  }];

  let arguments = (ins
    StrAttr:$sym_name,
    // TODO(benvanik): verify AnyRankedTensor.
    TypeAttr:$type,
    UnitAttr:$is_mutable,
    // TODO(benvanik): verify matches $type.
    OptionalAttr<FlatSymbolRefAttr>:$initializer,
    // TODO(benvanik): verify matches $type.
    OptionalAttr<AnyAttr>:$initial_value
  );

  let skipDefaultBuilders = 1;
  let builders = [
    OpBuilder<[{
      Builder *builder, OperationState &result, StringRef name, bool isMutable,
      FuncOp initializer, ArrayRef<NamedAttribute> attrs = {}
    }]>,
    OpBuilder<[{
      Builder *builder, OperationState &result, StringRef name, bool isMutable,
      Type type, Attribute initialValue, ArrayRef<NamedAttribute> attrs = {}
    }]>,
    OpBuilder<[{
      Builder *builder, OperationState &result, StringRef name, bool isMutable,
      Type type, ArrayRef<NamedAttribute> attrs = {}
    }]>,
  ];

  let verifier = [{ return verifyVariableOp(*this); }];

  let hasCanonicalizer = 1;
}

def FLOW_VariableLoadOp : FLOW_Op<"variable.load"> {
  let summary = [{loads a value from a global variable}];
  let description = [{
    Returns a copy of the variable value.
  }];

  let arguments = (ins
    FLOW_VariableRefAttr:$variable
  );
  let results = (outs
    AnyRankedTensor:$result
  );

  let verifier = [{ return verifyVariableLoadOp(*this); }];

  let hasCanonicalizer = 1;
}

def FLOW_VariableStoreOp : FLOW_Op<"variable.store"> {
  let summary = [{stores a value into a global variable}];
  let description = [{
    Stores a copy of the value into a variable.
  }];

  let arguments = (ins
    FLOW_VariableRefAttr:$variable,
    AnyRankedTensor:$value
  );

  let verifier = [{ return verifyVariableStoreOp(*this); }];

  let hasCanonicalizer = 1;
}

// TODO(benvanik): additional resource variable ops (like scatter/gather).

//===----------------------------------------------------------------------===//
// Partitioned regions
//===----------------------------------------------------------------------===//

def FLOW_DispatchRegionOp : FLOW_PureOp<"dispatch.region"> {
  let summary = [{partitioned region representing a dispatched workload}];
  let description = [{
    A closure that represents a functional dispatch unit. These perform
    computations in a way that can be lowered to target executable formats such
    as SPIR-V for execution.

    Ops that are identified as "dispatchable" are grouped into dispatch regions
    and compatible dispatch regions are folded together. What remains outside of
    the dispatch regions is the glue required to schedule the work (commonly
    referred to as "host" code, even if it doesn't run on an AP).

    Dispatch regions are modeled using value semantics: it is assumed that all
    arguments are read-only and that the dispatch regions themselves have no
    side-effects.
  }];

  let arguments = (ins
    FLOW_Workload:$workload,
    Variadic<AnyType>:$args
  );
  let results = (outs
    Variadic<AnyType>:$results
  );

  let regions = (region AnyRegion:$body);

  let extraClassDeclaration = [{
    /// Returns the index of the args() operand in the Operation operands list.
    unsigned mapArgOperandToOpOperand(unsigned i) { return i + 1; }
  }];

  let skipDefaultBuilders = 1;
  let builders = [
    OpBuilder<[{
      Builder *builder, OperationState &state, ArrayRef<Type> resultTypes,
      Value *workload, ArrayRef<Value *> args,
      ArrayRef<NamedAttribute> attributes = {}
    }]>,
  ];
}

def FLOW_ReductionRegionOp : FLOW_PureOp<"reduction.region", [
    SameVariadicOperandSize,
    // TODO(benvanik): verify operands and initial values have the same element
    // types (but NOT the same shapes).
  ]> {
  let summary = [{partitioned reduction region}];
  let description = [{
    A closure that defines a reduction operation over one or more inputs.
    Reductions are dispatches with very specific semantics around the indexing
    of work. Parititoning first isolates reduction regions prior to dispatch
    regions so that such semantics can be identified for folding.

    This operation follows the XLA Reduce semantics:
    https://www.tensorflow.org/xla/operation_semantics#reduce
  }];

  let arguments = (ins
    FLOW_Workload:$workload,
    Variadic<AnyType>:$operands,
    Variadic<AnyType>:$initial_values,
    // TODO(benvanik): use index types instead of i32.
    OptionalAttr<I32ElementsAttr>:$dimensions
  );
  let results = (outs
    Variadic<AnyType>:$results
  );

  let regions = (region AnyRegion:$body);

  let extraClassDeclaration = [{
    unsigned getNumReductionOperands() {
      return std::distance(operands().begin(), operands().end());
    }
  }];

  let skipDefaultBuilders = 1;
  let builders = [
    OpBuilder<[{
      Builder *builder, OperationState &state, ArrayRef<Type> resultTypes,
      Value *workload, ArrayRef<Value *> operands,
      ArrayRef<Value *> initialValues, ArrayRef<int32_t> dimensions,
      ArrayRef<NamedAttribute> attributes = {}
    }]>,
  ];
}

def FLOW_WindowedReductionRegionOp : FLOW_PureOp<"windowed_reduction.region", [
    SameVariadicOperandSize,
    // TODO(benvanik): verify operands and initial values have the same element
    // types (but NOT the same shapes).
  ]> {
  let summary = [{partitioned reduction region}];
  let description = [{
    A closure that defines a reduction operation over one or more inputs.
    Reductions are dispatches with very specific semantics around the indexing
    of work. Parititoning first isolates reduction regions prior to dispatch
    regions so that such semantics can be identified for folding.

    This operation follows the XLA ReduceWindow semantics:
    https://www.tensorflow.org/xla/operation_semantics#reducewindow
  }];

  let arguments = (ins
    FLOW_Workload:$workload,
    Variadic<AnyType>:$operands,
    Variadic<AnyType>:$initial_values,
    // TODO(benvanik): use index types instead of i32.
    I32ElementsAttr:$window_dimensions,
    I32ElementsAttr:$window_strides,
    I32ElementsAttr:$base_dilations,
    I32ElementsAttr:$window_dilations,
    FLOW_PaddingModeAttr:$padding_mode
  );
  let results = (outs
    Variadic<AnyType>:$results
  );

  let regions = (region AnyRegion:$body);

  let extraClassDeclaration = [{
    unsigned getNumReductionOperands() {
      return std::distance(operands().begin(), operands().end());
    }
  }];

  let skipDefaultBuilders = 1;
  let builders = [
    OpBuilder<[{
      Builder *builder, OperationState &state, ArrayRef<Type> resultTypes,
      Value *workload, ArrayRef<Value *> operands,
      ArrayRef<Value *> initialValues, ArrayRef<int32_t> windowDimensions,
      ArrayRef<int32_t> windowStrides, ArrayRef<int32_t> baseDilations,
      ArrayRef<int32_t> windowDilations, PaddingMode paddingMode,
      ArrayRef<NamedAttribute> attributes = {}
    }]>,
  ];
}

def FLOW_ReturnOp : FLOW_Op<"return", [Terminator]> {
  let summary = [{return from a flow.dispatch_region}];
  let description = [{
    Returns the given values from the region and back to the host code.
  }];

  let arguments = (ins
    Variadic<AnyType>:$operands
  );

  let builders = [
    OpBuilder<[{
      Builder *builder, OperationState &result
    }], [{
      build(builder, result, llvm::None);
    }]>,
  ];
}

//===----------------------------------------------------------------------===//
// Dispatch ops
//===----------------------------------------------------------------------===//

def FLOW_DispatchOp : FLOW_PureOp<"dispatch"> {
  let summary = [{a dispatch to an outlined dispatch region}];
  let description = [{
    Dispatches a workload to the specified executable function.
  }];

  let arguments = (ins
    // TODO(benvanik): replace with SymbolRefAttr.
    // TODO(benvanik): validate target is an executable.
    FlatSymbolRefAttr:$executable,
    FlatSymbolRefAttr:$entry_point,
    FLOW_Workload:$workload,
    Variadic<AnyType>:$operands
  );
  let results = (outs
    Variadic<AnyType>:$results
  );

  let skipDefaultBuilders = 1;
  let builders = [
    OpBuilder<[{
      Builder *builder, OperationState &result, StringRef executable,
      StringRef entryPoint, Value *workload,
      ArrayRef<Type> results, ArrayRef<Value *> operands = {}
    }], [{
      result.addOperands({workload});
      result.addOperands(operands);
      result.addAttribute("executable", builder->getSymbolRefAttr(executable));
      result.addAttribute("entry_point", builder->getSymbolRefAttr(entryPoint));
      result.addTypes(results);
    }]>,
  ];

  let extraClassDeclaration = [{
    FunctionType getEntryPointType();
  }];
}

//===----------------------------------------------------------------------===//
// Executables for outlined regions
//===----------------------------------------------------------------------===//

def FLOW_ExecutableOp : FLOW_Op<"executable", [
    IsolatedFromAbove,
    SingleBlockImplicitTerminator<"IREE::Flow::ExecutableEndOp">,
    NativeOpTrait<"SymbolTable">,
    Symbol,
  ]> {
  let summary = [{generic executable module}];
  let description = [{
    An executable module containing one or more public functions. The contents
    of the functions are safe to dispatch and can be lowered further to
    target-specific backend IR representations.
  }];

  let arguments = (ins
    StrAttr:$sym_name
    // TODO(benvanik): add compatibility and versioning attributes.
  );

  let regions = (region SizedRegion<1>:$body);

  let skipDefaultBuilders = 1;
  let builders = [
    OpBuilder<[{
      Builder *builder, OperationState &state, StringRef name
    }]>,
  ];

  let extraClassDeclaration = [{
    Block& getBlock() { return body().front(); }

    ::mlir::ModuleOp getInnerModule() {
      return *getBlock().getOps<::mlir::ModuleOp>().begin();
    }
  }];

  let verifier = [{ return verifyExecutableOp(*this); }];
}

def FLOW_ExecutableEndOp : FLOW_Op<"executable_end", [
    HasParent<"IREE::Flow::ExecutableOp">,
    Terminator,
  ]> {
  let summary = [{terminator pseudo-op for the executable op}];

  let parser = [{ return parseRegionEndOp(parser, &result); }];
  let printer = [{ return printRegionEndOp(p, *this); }];
}

def FLOW_DispatchEntryOp : FLOW_Op<"dispatch.entry", [
    HasParent<"IREE::Flow::ExecutableOp">,
    Symbol,
  ]> {
  let summary = [{defines an executable entry point for dispatch operations}];
  let description = [{
    Specifies an exported function with an externally-visible alias. Multiple
    exports can reference the same internal function.
  }];

  // TODO(benvanik): add a list of all used workloads.
  let arguments = (ins
    StrAttr:$sym_name,
    // TODO(benvanik): ref into child module.
    FlatSymbolRefAttr:$function_ref,
    OptionalAttr<FLOW_WorkloadAttr>:$workload,
    OptionalAttr<FLOW_WorkloadAttr>:$workgroup_size
  );
}

def FLOW_ReductionEntryOp : FLOW_Op<"reduction.entry", [
    HasParent<"IREE::Flow::ExecutableOp">,
    Symbol,
  ]> {
  let summary = [{defines an executable entry point for reduction operations}];
  let description = [{
    Specifies an exported function with an externally-visible alias. Multiple
    exports can reference the same internal function. The computation represents
    a reduction operation that has additional backend-specific semantics that
    need to be lowered.

    This operation follows the XLA Reduce semantics:
    https://www.tensorflow.org/xla/operation_semantics#reduce
  }];

  // TODO(benvanik): add a list of all used workloads.
  let arguments = (ins
    // TODO(benvanik): ref into child module.
    StrAttr:$sym_name,
    FlatSymbolRefAttr:$function_ref,
    FlatSymbolRefAttr:$apply_ref,
    I32Attr:$dimension
  );
}

def FLOW_WindowedReductionEntryOp : FLOW_Op<"windowed_reduction.entry", [
    HasParent<"IREE::Flow::ExecutableOp">,
    Symbol,
  ]> {
  let summary = [{defines an executable entry point for reduction operations}];
  let description = [{
    Specifies an exported function with an externally-visible alias. Multiple
    exports can reference the same internal function. The computation represents
    a reduction operation that has additional backend-specific semantics that
    need to be lowered.

    This operation follows the XLA ReduceWindow semantics:
    https://www.tensorflow.org/xla/operation_semantics#reducewindow
  }];

  // TODO(benvanik): add a list of all used workloads.
  let arguments = (ins
    // TODO(benvanik): ref into child module.
    StrAttr:$sym_name,
    FlatSymbolRefAttr:$function_ref,
    FlatSymbolRefAttr:$apply_ref,
    I32Attr:$window_dimension,
    I32Attr:$window_stride,
    I32Attr:$base_dilation,
    I32Attr:$window_dilation,
    FLOW_PaddingModeAttr:$padding
  );
}

//===----------------------------------------------------------------------===//
// Tensor ops
//===----------------------------------------------------------------------===//

// TODO(benvanik): tensor casts for widening/narrowing? or rely on std?
// TODO(benvanik): DynamicUpdateSlice-equivalent?
// TODO(benvanik): structured control flow (if we want it here).

#endif  // IREE_DIALECT_FLOW_OPS

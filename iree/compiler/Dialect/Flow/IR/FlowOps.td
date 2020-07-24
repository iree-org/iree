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

include "iree/compiler/Dialect/Flow/IR/FlowBase.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

class FLOW_PureOp<string mnemonic, list<OpTrait> traits = []> :
    FLOW_Op<mnemonic, !listconcat(traits, [NoSideEffect])>;

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
      OpBuilder &builder, OperationState &result, StringRef name,
      bool isMutable, FuncOp initializer, ArrayRef<NamedAttribute> attrs = {}
    }]>,
    OpBuilder<[{
      OpBuilder &builder, OperationState &result, StringRef name,
      bool isMutable, Type type, Attribute initialValue,
      ArrayRef<NamedAttribute> attrs = {}
    }]>,
    OpBuilder<[{
      OpBuilder &builder, OperationState &result, StringRef name, bool isMutable,
      Type type, ArrayRef<NamedAttribute> attrs = {}
    }]>,
  ];

  let verifier = [{ return verifyVariableOp(*this); }];

  let hasCanonicalizer = 1;
}

def FLOW_VariableAddressOp : FLOW_PureOp<"variable.address"> {
  let summary = [{returns an address reference to a variable}];
  let description = [{
    Returns the address of a variable as a typed reference. Can be used with the
    variable load and store indirect ops.
  }];

  let arguments = (ins
    FLOW_VariableRefAttr:$variable
  );
  let results = (outs
    FLOW_VariablePtr:$result
  );

  let assemblyFormat = "$variable attr-dict `:` type($result)";
}

def FLOW_VariableLoadOp : FLOW_Op<"variable.load", [MemoryEffects<[MemRead]>]> {
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

  let assemblyFormat = "$variable attr-dict `:` type($result)";

  let verifier = [{ return verifyVariableLoadOp(*this); }];

  let hasFolder = 1;
}

def FLOW_VariableLoadIndirectOp : FLOW_Op<"variable.load.indirect"> {
  let summary = [{loads a value from a global variable}];
  let description = [{
    Returns a copy of the variable value.
  }];

  let arguments = (ins
    FLOW_VariablePtr:$variable
  );
  let results = (outs
    AnyRankedTensor:$result
  );

  let assemblyFormat = "$variable attr-dict `:` type($variable) `->` type($result)";

  let verifier = [{ return verifyVariableLoadIndirectOp(*this); }];

  let hasCanonicalizer = 1;
}

def FLOW_VariableStoreOp : FLOW_Op<"variable.store"> {
  let summary = [{stores a value into a global variable}];
  let description = [{
    Stores a copy of the value into a variable.
  }];

  let arguments = (ins
    AnyRankedTensor:$value,
    FLOW_VariableRefAttr:$variable
  );

  let assemblyFormat = "$value `,` $variable attr-dict `:` type($value)";

  let verifier = [{ return verifyVariableStoreOp(*this); }];

  let hasCanonicalizer = 1;
}

def FLOW_VariableStoreIndirectOp : FLOW_Op<"variable.store.indirect"> {
  let summary = [{stores a value into a global variable}];
  let description = [{
    Stores a copy of the value into a variable.
  }];

  let arguments = (ins
    AnyRankedTensor:$value,
    FLOW_VariablePtr:$variable
  );

  let assemblyFormat = "$value `,` $variable attr-dict `:` type($value) `->` type($variable)";

  let verifier = [{ return verifyVariableStoreIndirectOp(*this); }];

  let hasCanonicalizer = 1;
}

// TODO(benvanik): additional resource variable ops (like scatter/gather).

//===----------------------------------------------------------------------===//
// Partitioned regions
//===----------------------------------------------------------------------===//

def FLOW_DispatchRegionOp : FLOW_PureOp<"dispatch.region", [
    IsolatedFromAbove,
  ]> {
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
    /// Forms a dispatch region around a given anchor operation, returning
    /// the new DispatchRegionOp and anchor operation within the region.
    /// Returns llvm::None on failure.
    /// The insertion point of the OpBuilder will be modified.
    static llvm::Optional<std::pair<DispatchRegionOp, Operation *>> 
        formFromAnchorOp(Value workload, Operation *anchorOp, 
                         OpBuilder &builder);

    /// Performs an in-place DCE optimization on unused operands and results.
    /// Note that this may or may not re-allocate the op. If so, the reference
    /// will be updated.
    static void dceOperandsAndResults(DispatchRegionOp &op);

    // Appends results to the dispatch region. This will re-allocate the
    // DispatchRegionOp itself but preserve the contained body block.
    // Returns a ResultRange for the new dispatch region op's results
    // corresponding to addlResults.
    static ResultRange appendResults(
        DispatchRegionOp &self, ValueRange addlResults, OpBuilder &builder);

    /// Returns the index of the args() operand in the Operation operands list.
    unsigned mapArgOperandToOpOperand(unsigned i) { return i + 1; }

    /// Inlines an op into the dispatch region.
    /// By default, this will inline the op at the beginning of the region.
    /// Set positionAtEnd=true to inline at the end. This is not a general
    /// IR splicing helper: it can only inline ops with inputs that map to
    /// either captured operands or results and is used to coelesce an op
    /// into an adjacent dispatch region.
    /// Note that the original op is cloned but not erased. It is up to the
    /// caller to cleanup the original op as needed.
    Operation *inlineOp(Operation *origOp, OpBuilder &builder, 
        bool positionAtEnd=false);
  }];

  let skipDefaultBuilders = 1;
  let builders = [
    OpBuilder<[{
      OpBuilder &builder, OperationState &state, ArrayRef<Type> resultTypes,
      Value workload, ValueRange args,
      ArrayRef<NamedAttribute> attributes = {}
    }]>,
  ];

  let hasCanonicalizer = 1;  
}

def FLOW_ReturnOp : FLOW_Op<"return", [Terminator]> {
  let summary = [{return from a flow.dispatch_region}];
  let description = [{
    Returns the given values from the region and back to the host code.
  }];

  let arguments = (ins
    Variadic<AnyType>:$operands
  );

  let assemblyFormat = "attr-dict ($operands^ `:` type($operands))?";

  let builders = [
    OpBuilder<[{
      OpBuilder &builder, OperationState &result
    }], [{
      build(builder, result, llvm::None);
    }]>,
  ];
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
      OpBuilder &builder, OperationState &state, StringRef name
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
  let assemblyFormat = "attr-dict";
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
    OptionalAttr<FLOW_WorkloadAttr>:$workload
  );
}

//===----------------------------------------------------------------------===//
// Dispatch ops
//===----------------------------------------------------------------------===//

def FLOW_DispatchOp : FLOW_PureOp<"dispatch", [
    FLOW_StreamableOp,
  ]> {
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
      OpBuilder &builder, OperationState &result, StringRef executable,
      StringRef entryPoint, Value workload,
      ArrayRef<Type> results, ValueRange operands = {}
    }], [{
      result.addOperands({workload});
      result.addOperands(operands);
      result.addAttribute("executable", builder.getSymbolRefAttr(executable));
      result.addAttribute("entry_point", builder.getSymbolRefAttr(entryPoint));
      result.addTypes(results);
    }]>,
  ];

  let extraClassDeclaration = [{
    FunctionType getEntryPointType();

    // StreamableOpInterface:
    bool isTransfer() { return false; }
    bool isUsableInStream() { return true; }
    bool isStreamOnly() { return true; }
  }];
}

//===----------------------------------------------------------------------===//
// Tensor ops
//===----------------------------------------------------------------------===//

def FLOW_TensorReshapeOp : FLOW_PureOp<"tensor.reshape", [
    FLOW_StreamableOp,
    AllElementTypesMatch<["source", "result"]>,
  ]> {
  let summary = [{reshapes a tensor}];
  let description = [{
    Reshapes a tensor to a new shape without modifying the contents.
  }];

  let arguments = (ins
    FLOW_Tensor:$source
    // TODO(benvanik): FLOW_Shape:$shape when supporting dynamic shapes.
  );
  let results = (outs
    FLOW_Tensor:$result
  );

  let assemblyFormat = "$source `:` type($source) `->` type($result) attr-dict";

  let extraClassDeclaration = [{
    // StreamableOpInterface:
    bool isTransfer() { return true; }
    bool isUsableInStream() { return true; }
    // TODO(benvanik): allow out of stream to act as a shape manipulation.
    bool isStreamOnly() { return true; }
  }];

  // TODO(benvanik): canonicalize away if resulting ops don't care.
  let hasFolder = 1;
}

def FLOW_TensorLoadOp : FLOW_PureOp<"tensor.load", [
    TypesMatchWith<"value type matches element type of target operand",
                   "source", "result",
                   "$_self.cast<ShapedType>().getElementType()">,
  ]> {
  let summary = [{loads a value from a tensor element}];
  let description = [{
    Returns the element at the given location from within the tensor.
  }];

  let arguments = (ins
    FLOW_Tensor:$source,
    Variadic<FLOW_Dim>:$indices
  );
  let results = (outs
    AnyTypeOf<[FLOW_PrimitiveType, AnyVector]>:$result
  );

  let assemblyFormat = [{
    $source (`[` $indices^ `]`)? `:` type($source) attr-dict-with-keyword
  }];

  // TODO(benvanik): canonicalize to slice+load if dims are known.
  let hasFolder = 1;
}

def FLOW_TensorStoreOp : FLOW_PureOp<"tensor.store", [
    AllTypesMatch<["target", "result"]>,
    TypesMatchWith<"value type matches element type of target operand",
                   "target", "value",
                   "$_self.cast<ShapedType>().getElementType()">,
  ]> {
  let summary = [{stores a value into a tensor element}];
  let description = [{
    Returns a tensor with the element at the given index set to the given value.
  }];

  let arguments = (ins
    AnyTypeOf<[FLOW_PrimitiveType, AnyVector]>:$value,
    FLOW_Tensor:$target,
    Variadic<FLOW_Dim>:$indices
  );
  let results = (outs
    FLOW_Tensor:$result
  );

  let assemblyFormat = [{
    $value `,` $target (`[` $indices^ `]`)? `:` type($target)
    attr-dict-with-keyword
  }];

  let hasFolder = 1;
}

def FLOW_TensorSplatOp : FLOW_PureOp<"tensor.splat", [
    FLOW_StreamableOp,
    TypesMatchWith<"value type matches element type of result",
                   "result", "value",
                   "$_self.cast<ShapedType>().getElementType()">,
  ]> {
  let summary = [{splats a value into a shaped tensor}];
  let description = [{
    Returns a tensor initialized to the given primitive value.
  }];

  let arguments = (ins
    FLOW_PrimitiveType:$value
    // TODO(benvanik): FLOW_Shape:$shape when supporting dynamic shapes.
  );
  let results = (outs
    FLOW_Tensor:$result
  );

  let assemblyFormat = "$value `:` type($result) attr-dict-with-keyword";

  let extraClassDeclaration = [{
    // StreamableOpInterface:
    bool isTransfer() { return true; }
    bool isUsableInStream() { return true; }
    // TODO(benvanik): allow out of stream to act as a hal.buffer.fill.
    bool isStreamOnly() { return true; }
  }];

  // TODO(benvanik): canonicalize splat+slice to smaller splat.
  let hasFolder = 1;
}

def FLOW_TensorCloneOp : FLOW_PureOp<"tensor.clone", [
    FLOW_StreamableOp,
    SameOperandsAndResultType,
  ]> {
  let summary = [{performs a full tensor clone operation}];
  let description = [{
    Clones the input tensor into an identical output tensor.
  }];

  let arguments = (ins
    FLOW_Tensor:$operand
  );
  let results = (outs
    FLOW_Tensor:$result
  );

  let assemblyFormat = "$operand `:` type($result) attr-dict";

  let extraClassDeclaration = [{
    // StreamableOpInterface:
    bool isTransfer() { return true; }
    bool isUsableInStream() { return true; }
    // TODO(benvanik): allow out of stream to act as a hal.buffer.copy.
    bool isStreamOnly() { return true; }
  }];

  // TODO(benvanik): canonicalize away entirely in most cases.
  let hasFolder = 1;
}

def FLOW_TensorSliceOp : FLOW_PureOp<"tensor.slice", [
    FLOW_StreamableOp,
    AllRanksMatch<["source", "result"]>,
    AllElementTypesMatch<["source", "result"]>,
    SameVariadicOperandSize,
  ]> {
  let summary = [{slices out a subregion of a tensor}];
  let description = [{
    Clones a subregion of a tensor.
  }];

  let arguments = (ins
    FLOW_Tensor:$source,
    Variadic<FLOW_Dim>:$start_indices,
    Variadic<FLOW_Dim>:$lengths
    // TODO(benvanik): strides.
  );
  let results = (outs
    FLOW_Tensor:$result
  );

  let assemblyFormat = [{
    $source `[` $start_indices `for` $lengths `]` `:` type($source) `->`
    type($result) attr-dict
  }];

  let extraClassDeclaration = [{
    // StreamableOpInterface:
    bool isTransfer() { return true; }
    bool isUsableInStream() { return true; }
    // TODO(benvanik): allow out of stream to act as a hal.buffer.slice.
    bool isStreamOnly() { return true; }
  }];

  // TODO(benvanik): canonicalize multiple slices (traverse upward through ssa).
  let hasFolder = 1;
}

def FLOW_TensorUpdateOp : FLOW_PureOp<"tensor.update", [
    FLOW_StreamableOp,
    AllRanksMatch<["update", "target", "result"]>,
    AllTypesMatch<["target", "result"]>,
    AllElementTypesMatch<["update", "target", "result"]>,
  ]> {
  let summary = [{updates a tensor with the contents of another tensor}];
  let description = [{
    Updates the target tensor with the contents of the update tensor at the
    given offset indices.
  }];

  let arguments = (ins
    FLOW_Tensor:$update,
    FLOW_Tensor:$target,
    Variadic<FLOW_Dim>:$start_indices
  );
  let results = (outs
    FLOW_Tensor:$result
  );

  let assemblyFormat = [{
    $update `,` $target `[` $start_indices `]` `:` type($update) `->`
    type($result) attr-dict
  }];

  let extraClassDeclaration = [{
    // StreamableOpInterface:
    bool isTransfer() { return true; }
    bool isUsableInStream() { return true; }
    // TODO(benvanik): allow out of stream to act as a hal.buffer.copy.
    bool isStreamOnly() { return true; }
  }];

  // TODO(benvanik): canonicalize contiguous updates/across slices.
  let hasFolder = 1;
}

def FLOW_TensorTraceOp : FLOW_Op<"tensor.trace", []> {
  let summary = [{trace value(s) operation}];
  let description = [{
    Trace point for dispatchable functions.
  }];

  let arguments = (ins Variadic<FLOW_Tensor>:$operands);

  let assemblyFormat = "attr-dict ($operands^ `:` type($operands))?";
}

//===----------------------------------------------------------------------===//
// Streams
//===----------------------------------------------------------------------===//

// TODO(benvanik): replace with real segmented stream ops.
def FLOW_ExStreamFragmentOp : FLOW_PureOp<"ex.stream.fragment", [
    IsolatedFromAbove,
  ]> {
  let summary = [{experimental op for defining formed stream regions}];
  let description = [{
    Represents a region where all of the dispatches are meant to target the
    same execution stream. This will be replaced with a segmented verison.
  }];

  let arguments = (ins
    Variadic<AnyType>:$args
  );
  let results = (outs
    Variadic<AnyType>:$results
  );

  let regions = (region AnyRegion:$body);

  let skipDefaultBuilders = 1;
  let builders = [
    OpBuilder<[{
      OpBuilder &builder, OperationState &state, ArrayRef<Type> resultTypes,
      ValueRange args, ArrayRef<NamedAttribute> attributes = {}
    }]>,
  ];

  let hasCanonicalizer = 1;
}

#endif  // IREE_DIALECT_FLOW_OPS

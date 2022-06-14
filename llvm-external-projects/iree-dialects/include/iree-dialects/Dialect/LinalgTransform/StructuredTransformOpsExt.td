// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef STRUCTURED_TRANSFORM_OPS_EXT
#define STRUCTURED_TRANSFORM_OPS_EXT

include "mlir/Dialect/PDL/IR/PDLTypes.td"
include "mlir/Dialect/Transform/IR/TransformDialect.td"
include "mlir/Dialect/Transform/IR/TransformInterfaces.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/OpAsmInterface.td"
include "mlir/IR/OpBase.td"

def FunctionalStyleMultiOperandMultiResultTransformOpTrait
    : NativeOpTrait<"FunctionalStyleMultiOperandMultiResultTransformOpTrait"> {
  let cppNamespace = "::mlir::transform";
}

def CanonicalizedSequenceOp
    : TransformDialectOp<"structured.canonicalized_sequence",
      [DeclareOpInterfaceMethods<RegionBranchOpInterface,
          ["getSuccessorEntryOperands", "getSuccessorRegions",
           "getRegionInvocationBounds"]>,
       DeclareOpInterfaceMethods<TransformOpInterface>,
       DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
       OpAsmOpInterface, PossibleTopLevelTransformOpTrait,
       SingleBlockImplicitTerminator<"::mlir::transform::YieldOp">]> {

  let summary = "A transformation sequence interspersed with canonicalizations";
  let description = [{
    This op is a copy of `transform.sequence`, but applies canonicalizations
    after each step in the sequence.
  }];

  let arguments = (ins Optional<PDL_Operation>:$root);
  let results = (outs Variadic<AnyType>:$results);
  let regions = (region SizedRegion<1>:$body);

  let assemblyFormat =
    "($root^)? attr-dict-with-keyword regions (`:` type($results)^)?";

  let extraClassDeclaration = [{
    /// Allow the dialect prefix to be omitted.
    static ::llvm::StringRef getDefaultDialect() { return "transform"; }
  }];

  let cppNamespace = "transform_ext";
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//

def BufferizeOp : Op<Transform_Dialect, "bufferize",
    [DeclareOpInterfaceMethods<TransformOpInterface>, 
     FunctionalStyleMultiOperandMultiResultTransformOpTrait,
     MemoryEffectsOpInterface,
     TransformOpInterface]> {
  let description = [{Indicates that the entire module should be bufferized.}];
  let assemblyFormat = "attr-dict";
  let cppNamespace = "transform_ext";
}

def LowerVectorsOp : Op<Transform_Dialect, "lower_vectors",
    [DeclareOpInterfaceMethods<TransformOpInterface>, 
     FunctionalStyleMultiOperandMultiResultTransformOpTrait,
     MemoryEffectsOpInterface]> {
  let description = [{Indicates that the vector operations in the entire
  module should be lowered to simpler primitives (multiple stages of lowering
  be executed at once).}];

  let arguments =
    (ins DefaultValuedAttr<I64ArrayAttr, "{0, 1, 2, 3, 4, 5, 6}">:$stages,
     DefaultValuedAttr<StrAttr, "\"outerproduct\"">:$contraction_lowering,
     DefaultValuedAttr<StrAttr, "\"innerparallel\"">:$multireduction_lowering,
     DefaultValuedAttr<StrAttr, "\"linalg-copy\"">:$split_transfers,
     DefaultValuedAttr<BoolAttr, "true">:$unroll_vector_transfers,
     DefaultValuedAttr<StrAttr, "\"eltwise\"">:$transpose_lowering,
     DefaultValuedAttr<BoolAttr, "false">:$transpose_avx2_lowering
    );

  let assemblyFormat = "attr-dict";
  let cppNamespace = "transform_ext";
}

def LowerToLLVMOp : Op<Transform_Dialect, "lower_to_llvm",
    [DeclareOpInterfaceMethods<TransformOpInterface>,
     FunctionalStyleMultiOperandMultiResultTransformOpTrait,
     MemoryEffectsOpInterface]> {
  let description = [{Indicates that the entire module should be converted
  to the LLVM dialect. This is expected to be the last transformation in
  a sequence.}];

  let arguments =
    (ins DefaultValuedAttr<BoolAttr, "false">:$reassociate_fp_reductions,
     DefaultValuedAttr<BoolAttr, "false">:$enable_index_optimizations,
     DefaultValuedAttr<BoolAttr, "false">:$enable_arm_neon,
     DefaultValuedAttr<BoolAttr, "false">:$enable_arm_sve,
     DefaultValuedAttr<BoolAttr, "false">:$enable_amx,
     DefaultValuedAttr<BoolAttr, "false">:$enable_x86vector,
     DefaultValuedAttr<BoolAttr, "false">:$enable_async);

  let assemblyFormat = "attr-dict";
  let cppNamespace = "transform_ext";
}

def PrintOp : Op<Transform_Dialect, "print",
    [DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
     DeclareOpInterfaceMethods<TransformOpInterface>]> {
  let arguments = (ins Optional<PDL_Operation>:$target,
                       StrAttr:$name);
  let description = [{Prints the module.}];
  let assemblyFormat = "($target^)? attr-dict";
  let cppNamespace = "transform_ext";
}

#endif // STRUCTURED_TRANSFORM_OPS_EXT

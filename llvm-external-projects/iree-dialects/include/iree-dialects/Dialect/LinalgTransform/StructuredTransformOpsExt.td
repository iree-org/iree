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
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/OpAsmInterface.td"
include "mlir/IR/OpBase.td"

def FunctionalStyleMultiOperandMultiResultTransformOpTrait
    : NativeOpTrait<"FunctionalStyleMultiOperandMultiResultTransformOpTrait"> {
  let cppNamespace = "::mlir::transform";
}

def CanonicalizedSequenceOp
    : Op<Transform_Dialect, "structured.canonicalized_sequence", [
        DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
        DeclareOpInterfaceMethods<TransformOpInterface>,
        OpAsmOpInterface,
        PossibleTopLevelTransformOpTrait,
        SingleBlockImplicitTerminator<"::mlir::transform::YieldOp">]> {
  let summary = "A transformation sequence interspersed with canonicalizations";

  let arguments = (ins Optional<PDL_Operation>:$target);
  let regions = (region SizedRegion<1>:$body);

  let assemblyFormat = "($target^)? attr-dict-with-keyword regions";
  let extraClassDeclaration = [{
    /// Allow the dialect prefix to be omitted.
    static ::llvm::StringRef getDefaultDialect() { return "transform"; }
  }];

  let cppNamespace = "transform_ext";
}

//===----------------------------------------------------------------------===//

def FuseOp : Op<Transform_Dialect, "structured.fuse",
    [FunctionalStyleMultiOperandMultiResultTransformOpTrait, 
     MemoryEffectsOpInterface,
     DeclareOpInterfaceMethods<TransformOpInterface>]> {
  let description = [{Tiles the operations pointed to by the target handle and
  fuses their producers greedily using the options provided as attributes.}];

  let arguments =
    (ins PDL_Operation:$target,
         DefaultValuedAttr<I64ArrayAttr, "{}">:$tile_sizes,
         DefaultValuedAttr<I64ArrayAttr, "{}">:$tile_interchange);
  let results = (outs PDL_Operation:$transformed,
                      Variadic<PDL_Operation>:$loops);

  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;
  let cppNamespace = "transform_ext";
}

def GeneralizeOp : Op<Transform_Dialect, "structured2.generalize",
    [FunctionalStyleTransformOpTrait, 
     MemoryEffectsOpInterface,
     TransformOpInterface,
     TransformEachOpTrait]> {
  let description = [{Generalizes the operations pointed to
  by the target handle.}];

  let arguments = (ins PDL_Operation:$target);
  let results = (outs PDL_Operation:$transformed);

  let assemblyFormat = "$target attr-dict";
  let cppNamespace = "transform_ext";

  let extraClassDeclaration = [{
    ::mlir::FailureOr<::mlir::linalg::LinalgOp> applyToOne(
        ::mlir::linalg::LinalgOp target);
  }];
}

def BufferizeOp : Op<Transform_Dialect, "bufferize",
    [DeclareOpInterfaceMethods<TransformOpInterface>, 
     FunctionalStyleMultiOperandMultiResultTransformOpTrait,
     MemoryEffectsOpInterface,
     TransformOpInterface]> {
  let description = [{Indicates that the entire module should be bufferized.}];
  let assemblyFormat = "attr-dict";
  let cppNamespace = "transform_ext";
}

def DecomposeOp : Op<Transform_Dialect, "structured2.decompose",
    [DeclareOpInterfaceMethods<TransformOpInterface>,
     MemoryEffectsOpInterface,
     FunctionalStyleMultiOperandMultiResultTransformOpTrait]> {
  let description = [{Indicates that ops in the entire module should be
  decomposed into lower-level components.}];
  let assemblyFormat = "attr-dict";
  let cppNamespace = "transform_ext";
}

def VectorizeOp : Op<Transform_Dialect, "structured2.vectorize",
     [DeclareOpInterfaceMethods<TransformOpInterface>,
      FunctionalStyleMultiOperandMultiResultTransformOpTrait,
      MemoryEffectsOpInterface]> {
  let description = [{Indiactes that vectorization should be performed. If a
  target handle is provided, only vectorizes the operations pointed to by the
  handle. Otherwise vectorizes the entire module. Vectorization options are
  provided as operation attributes.}];

  let arguments = (ins Optional<PDL_Operation>:$target,
                   DefaultValuedAttr<BoolAttr, "false">:$vectorize_padding
                  );
  let results = (outs Optional<PDL_Operation>:$transformed);

  let hasCustomAssemblyFormat = 1;
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

def GetParentLoopOp : Op<Transform_Dialect, "get_parent_loop",
    [DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
     FunctionalStyleTransformOpTrait,
     TransformEachOpTrait,
     TransformOpInterface]> {
  let description = [{Obtains a handle to a parent loop of the given
  operation.}];

  let arguments =
    (ins PDL_Operation:$target,
         DefaultValuedAttr<Confined<I64Attr, [IntPositive]>,
                           "1">:$num_loops);
  let results = (outs PDL_Operation:$transformed);

  let assemblyFormat = "$target attr-dict";
  let cppNamespace = "transform_ext";

  let extraClassDeclaration = [{
    ::mlir::FailureOr<::mlir::scf::ForOp> applyToOne(::mlir::Operation *source);
  }];
}

def UnrollLoopOp : Op<Transform_Dialect, "unroll_loop",
    [FunctionalStyleMultiOperandMultiResultTransformOpTrait,
     MemoryEffectsOpInterface,
     TransformEachOpTrait,
     TransformOpInterface]> {
  let description = [{Unrolls the given loop with the given unroll factor.}];

  let arguments = (ins PDL_Operation:$target,
                       Confined<I64Attr, [IntPositive]>:$factor);

  let assemblyFormat = "$target attr-dict";
  let cppNamespace = "transform_ext";

  let extraClassDeclaration = [{
    ::mlir::LogicalResult applyToOne(::mlir::scf::ForOp loop);
  }];
}

def PeelLoopOp : Op<Transform_Dialect, "peel_loop",
    [FunctionalStyleTransformOpTrait,
     MemoryEffectsOpInterface,
     TransformEachOpTrait,
     TransformOpInterface]> {
  let arguments = (ins PDL_Operation:$target);
  let results = (outs PDL_Operation:$transformed);

  let assemblyFormat = "$target attr-dict";
  let cppNamespace = "transform_ext";

  let extraClassDeclaration = [{
    ::mlir::FailureOr<::mlir::scf::ForOp> applyToOne(::mlir::scf::ForOp loop);
  }];
}

def PipelineLoopOp : Op<Transform_Dialect, "pipeline_loop",
    [FunctionalStyleTransformOpTrait,
     MemoryEffectsOpInterface,
     TransformEachOpTrait,
     TransformOpInterface]> {
  let arguments = (ins PDL_Operation:$target,
                   DefaultValuedAttr<I64Attr, "1">:$iteration_interval,
                   DefaultValuedAttr<I64Attr, "10">:$read_latency);
  let results = (outs PDL_Operation:$transformed);

  let assemblyFormat = "$target attr-dict";
  let cppNamespace = "transform_ext";

  let extraClassDeclaration = [{
    ::mlir::FailureOr<::mlir::scf::ForOp> applyToOne(::mlir::scf::ForOp loop);
  }];
}

def OutlineLoopOp : Op<Transform_Dialect, "outline_loop",
    [FunctionalStyleTransformOpTrait,
     MemoryEffectsOpInterface,
     DeclareOpInterfaceMethods<TransformOpInterface>]> {
  let arguments = (ins PDL_Operation:$target,
                   StrAttr:$func_name);
  let results = (outs PDL_Operation:$transformed);

  let assemblyFormat = "$target attr-dict";
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

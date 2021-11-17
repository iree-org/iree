// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_CODEGEN_DIALECT_LOWERINGCONFIG
#define IREE_COMPILER_CODEGEN_DIALECT_LOWERINGCONFIG

include "iree/compiler/Codegen/Dialect/IREECodegenDialect.td"

// List of pre-existing pipelines for translating executables.
def CPU_Default
    : StrEnumAttrCase<"CPUDefault">;
def CPU_TensorToVectors
    : StrEnumAttrCase<"CPUTensorToVectors">;
def CPU_TileFuseAndVectorize
    : StrEnumAttrCase<"CPUTileFuseAndVectorize">;

def LLVMGPU_SimpleDistribute
    : StrEnumAttrCase<"LLVMGPUDistribute">;
def LLVMGPU_Vectorize
    : StrEnumAttrCase<"LLVMGPUVectorize">;
def LLVMGPU_MatmulSimt
    : StrEnumAttrCase<"LLVMGPUMatmulSimt">;
def LLVMGPU_MatmulTensorCore
    : StrEnumAttrCase<"LLVMGPUMatmulTensorCore">;

def SPIRV_SimpleDistribute
    : StrEnumAttrCase<"SPIRVDistribute">;
def SPIRV_Vectorize
    : StrEnumAttrCase<"SPIRVVectorize">;
def SPIRV_VectorizeToCooperativeOps
    : StrEnumAttrCase<"SPIRVVectorizeToCooperativeOps">;

def None
    : StrEnumAttrCase<"None">;

// EnumAttrCase for all known lowerings for ops within dispatch region
// to scalar/native-vector code.
def DispatchLoweringPassPipelineEnum : StrEnumAttr<
    "DispatchLoweringPassPipeline",
    "identifier for pass pipeline use to lower dispatch region",
    [CPU_Default, CPU_TensorToVectors, CPU_TileFuseAndVectorize,
     LLVMGPU_SimpleDistribute, LLVMGPU_Vectorize, LLVMGPU_MatmulSimt,
     LLVMGPU_MatmulTensorCore, SPIRV_SimpleDistribute, SPIRV_Vectorize,
     SPIRV_VectorizeToCooperativeOps, None]> {
  let cppNamespace = "::mlir::iree_compiler::IREE::Codegen";
}

def IREECodegen_TranslationInfoAttr :
    AttrDef<IREECodegen_Dialect, "TranslationInfo", []> {
  let mnemonic = "translation.info";
  let summary = [{drive dispatch entry point lowering}];
  let description = [{
    Specifies the information that is used to drive the translation of
    an entry point function using Linalg based structured-op
    lowering.. During executable translation this is attached to the
    `hal.executable.entry_point` operation.

    If this operation is already set on the root operation (as part of
    `iree_codegen.compilation.info`) that drives the compilation of a
    dispatch region (like `linalg.matmul`/`linalg.*conv*`), this
    attribute gets propagated to the entry point function.

    The fields are
    - `passPipeline` : The pass pipeline to use.
    - `workloadPerWorkgroup` : Specifies how much of the original
      `workload` is handled by a workgroup along `x`, `y` and `z`.  If
      left empty it implies that that there is a single workgroup that
      does the entire `workload`.

  }];

  let assemblyFormat = [{
    `<` $passPipeline `,` `workload_per_wg` `=` $workloadPerWorkgroup `>`
  }];

  let parameters = (ins
    AttrParameter<"StringAttr", "">:$passPipeline,
    AttrParameter<"ArrayAttr", "">:$workloadPerWorkgroup
  );
  let builders = [
    AttrBuilder<(ins "DispatchLoweringPassPipeline":$passPipeline,
        CArg<"ArrayRef<int64_t>", "{}">:$workloadPerWorkgroup)>
  ];
  let extraClassDeclaration = [{
    // Returns the lowering pass pipeline set.
    DispatchLoweringPassPipeline getDispatchLoweringPassPipeline();

    // Returns values of the workloadPerWorkgroup field if set.
    SmallVector<int64_t> getWorkloadPerWorkgroupVals();
  }];
  let genVerifyDecl = 1;
}

def IREECodegen_LoweringConfigAttr :
    AttrDef<IREECodegen_Dialect, "LoweringConfig", []> {
  let mnemonic = "lowering.config";
  let summary = [{drive lowering of an operation within dispatch region}];
  let description = [{
    Specifies the information that is used by backend compiler to
    translate an operation to scalar code. The way the information is
    used is specific to each backend (indeed specific to the pass
    pipeline used) to compile that operation.

    TODO: Currently there is no verification that the configuration
    specifies everything needed for a pass-pipeline. The values to set
    for these parameters is dependent on the pass-pipeline
    implementation. In future, each pass pipeline could verify that
    the lowering configuration has all the necessary attributes for
    the pipeline.

  }];

  let assemblyFormat = [{
    `<` `tile_sizes` `=` $tileSizes `,` `native_vector_size` `=` $nativeVectorSize `>`
  }];

  let parameters = (ins
    AttrParameter<"ArrayAttr", "">:$tileSizes,
    AttrParameter<"ArrayAttr", "">:$nativeVectorSize
  );
  let builders = [
    AttrBuilder<(ins "TileSizesListTypeRef":$tileSizes,
        CArg<"ArrayRef<int64_t>", "{}">:$nativeVectorSize)>
  ];
  let extraClassDeclaration = [{
    // Returns the tile sizes for all levels set for the op.
    TileSizesListType getTileSizeVals();

    // Returns the tile sizes for a level set for the op.
    SmallVector<int64_t> getTileSizeVals(unsigned level = 0);

    // Returns the native vector size to use.
    SmallVector<int64_t> getNativeVectorSizeVals();
  }];
  let genVerifyDecl = 1;
}

def IREECodegen_CompilationInfoAttr :
    AttrDef<IREECodegen_Dialect, "CompilationInfo", []> {
  let mnemonic = "compilation.info";
  let summary = [{drive lowering of an operation from input dialect}];
  let description = [{
    Specifies the information that allows controlling the compilation
    of operations like `linalg.matmul`/`linalg.*conv` within
    IREE. This information is used to override the defaults used by
    the IREE compiler. Currently it is only valid to set this on
    `linalg.matmul`/`linalg.*conv*` operations.

    TODO: It is expected that the `TranslationInfoAttr` and the
    `LoweringConfigAttr` are specified. Currently there is no
    verification that the values of the `LoweringConfigAttr` fully
    specifies the behaviour of the compilation path chosen with
    `TranslationInfoAttr`. This could be added in the future.  Note:
    Typically the values used for the first-level tiling in
    `LoweringConfigAttr` and `workload_per_wg` value in the
    `TranslationInfoAttr` are the same since the first-level of tile +
    distribute is already done at the `Flow` level. This verification
    is also a TODO.
  }];
  let parameters = (ins
    AttrParameter<"LoweringConfigAttr", "">:$loweringConfig,
    AttrParameter<"TranslationInfoAttr", "">:$translationInfo,
    AttrParameter<"ArrayAttr", "">:$workgroupSize
  );

  let assemblyFormat = [{
    `<` $loweringConfig `,` $translationInfo `,` `workgroup_size` `=` $workgroupSize `>`
  }];

  let builders = [
    AttrBuilder<(ins "TileSizesListTypeRef":$tileSizes,
      "ArrayRef<int64_t>":$nativeVectorSize,
      CArg<"ArrayRef<int64_t>", "{}">:$workgroupSize)>,
    AttrBuilder<(ins "TileSizesListTypeRef":$tileSizes,
      "ArrayRef<int64_t>":$nativeVectorSize,
      "DispatchLoweringPassPipeline":$passPipeline,
      "ArrayRef<int64_t>":$workloadPerWorkgroup,
      CArg<"ArrayRef<int64_t>", "{}">:$workgroupSize)>,
  ];
  let extraClassDeclaration = [{
    SmallVector<int64_t> getWorkgroupSizeVals();
  }];
  let genVerifyDecl = 1;
}

#endif // IREE_COMPILER_CODEGEN_DIALECT_LOWERINGCONFIG

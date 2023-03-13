// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_CODEGEN_DIALECT_LOWERINGCONFIG
#define IREE_COMPILER_CODEGEN_DIALECT_LOWERINGCONFIG

include "iree/compiler/Codegen/Dialect/IREECodegenDialect.td"
include "mlir/IR/EnumAttr.td"

// List of pre-existing pipelines for translating executables.
def CPU_Default
    : I32EnumAttrCase<"CPUDefault", 0>;
def CPU_DoubleTilingExpert
    : I32EnumAttrCase<"CPUDoubleTilingExpert", 1>;
def CPU_DoubleTilingPadExpert
    : I32EnumAttrCase<"CPUDoubleTilingPadExpert", 2>;
def CPU_DoubleTilingPeelingExpert
    : I32EnumAttrCase<"CPUDoubleTilingPeelingExpert", 3>;
def CPU_ConvTileAndDecomposeExpert
    : I32EnumAttrCase<"CPUConvTileAndDecomposeExpert", 4>;
def CPU_Mmt4dTilingExpert
    : I32EnumAttrCase<"Mmt4dTilingExpert", 5>;
def CPU_BufferOpsTileAndVectorize
    : I32EnumAttrCase<"CPUBufferOpsTileAndVectorize", 6>;
def CPU_TripleTilingExpert : I32EnumAttrCase<"CPUTripleTilingExpert", 7>;

def CPU_DataTiling
    : I32EnumAttrCase<"CPUDataTiling", 8>;

def LLVMGPU_SimpleDistribute : I32EnumAttrCase<"LLVMGPUDistribute", 9>;
def LLVMGPU_Vectorize : I32EnumAttrCase<"LLVMGPUVectorize", 10>;
def LLVMGPU_MatmulSimt : I32EnumAttrCase<"LLVMGPUMatmulSimt", 11>;
def LLVMGPU_MatmulTensorCore : I32EnumAttrCase<"LLVMGPUMatmulTensorCore", 12>;
def LLVMGPU_TransposeSharedMem : I32EnumAttrCase<"LLVMGPUTransposeSharedMem", 13>;
def LLVMGPU_WarpReduction : I32EnumAttrCase<"LLVMGPUWarpReduction", 14>;
def LLVMGPU_PackUnPack : I32EnumAttrCase<"LLVMGPUPackUnPack", 15>;
def LLVMGPU_MatmulTensorCoreMmaSync : I32EnumAttrCase<"LLVMGPUMatmulTensorCoreMmaSync", 16>;

def SPIRV_BaseDistribute
    : I32EnumAttrCase<"SPIRVBaseDistribute", 17>;
def SPIRV_BaseVectorize
    : I32EnumAttrCase<"SPIRVBaseVectorize", 18>;
def SPIRV_MatmulPromoteVectorize
    : I32EnumAttrCase<"SPIRVMatmulPromoteVectorize", 19>;
def SPIRV_CooperativeMatrixVectorize
    : I32EnumAttrCase<"SPIRVCooperativeMatrixVectorize", 20>;
def SPIRV_SubgroupReduce
    : I32EnumAttrCase<"SPIRVSubgroupReduce", 21>;
def SPIRV_WinogradVectorize
    : I32EnumAttrCase<"SPIRVWinogradVectorize", 22>;

def VMVX_Default : I32EnumAttrCase<"VMVXDefault", 23>;


def Linalg_TransformDialectCodegen
    : I32EnumAttrCase<"TransformDialectCodegen", 100>;

def None : I32EnumAttrCase<"None", 0xff>;

// EnumAttrCase for all known lowerings for ops within dispatch region
// to scalar/native-vector code.
def DispatchLoweringPassPipelineEnum
    : I32EnumAttr<
          "DispatchLoweringPassPipeline",
          "identifier for pass pipeline use to lower dispatch region", [
            CPU_Default, CPU_DoubleTilingExpert, CPU_DoubleTilingPadExpert,
            CPU_DoubleTilingPeelingExpert, CPU_ConvTileAndDecomposeExpert,
            CPU_Mmt4dTilingExpert, CPU_BufferOpsTileAndVectorize,
            CPU_TripleTilingExpert, CPU_DataTiling, LLVMGPU_SimpleDistribute,
            LLVMGPU_Vectorize, LLVMGPU_MatmulSimt, LLVMGPU_MatmulTensorCore,
            LLVMGPU_TransposeSharedMem, LLVMGPU_WarpReduction,
            LLVMGPU_PackUnPack, LLVMGPU_MatmulTensorCoreMmaSync,
            SPIRV_BaseDistribute, SPIRV_BaseVectorize,
            SPIRV_MatmulPromoteVectorize, SPIRV_CooperativeMatrixVectorize,
            SPIRV_SubgroupReduce, SPIRV_WinogradVectorize, VMVX_Default,
            // Transform dialect based codegen
            Linalg_TransformDialectCodegen, None
          ]> {
  let cppNamespace = "::mlir::iree_compiler::IREE::Codegen";
  // Don't generate a C++ class! We want to use the AttrDef
  let genSpecializedAttr = 0;
}

// Define the AttrDef
def DispatchLoweringPassPipelineAttr :
    EnumAttr<IREECodegen_Dialect, DispatchLoweringPassPipelineEnum, ""> {
  let assemblyFormat = "``$value";
}


def IREECodegen_TranslationInfoAttr :
    AttrDef<IREECodegen_Dialect, "TranslationInfo", []> {
  let mnemonic = "translation_info";
  let summary = [{drive dispatch entry point lowering}];
  let description = [{
    Specifies the information that is used to drive the translation of
    an entry point function using Linalg based structured-op
    lowering.. During executable translation this is attached to the
    `hal.executable.export` operation.

    If this operation is already set on the root operation (as part of
    `iree_codegen.compilation_info`) that drives the compilation of a
    dispatch region (like `linalg.matmul`/`linalg.*conv*`), this
    attribute gets propagated to the entry point function.

    The fields are
    - `passPipeline` : The pass pipeline to use.

  }];

  let assemblyFormat = [{
    `<` `` $passPipeline
    (`pipeline_depth` `=` $softwarePipelineDepth^)?
    (`store_stage` `=` $softwarePipelineStoreStage^)? `>`
  }];

  let parameters = (ins
    AttrParameter<"IREE::Codegen::DispatchLoweringPassPipelineAttr",
        "Name of the pipeline to be invoked on the translation unit.">:$passPipeline,
    OptionalParameter<"unsigned",
        "The software pipeline depth to be used">:$softwarePipelineDepth,
    DefaultValuedParameter<"unsigned", "1",
        "The software pipeline stage to place stores">:$softwarePipelineStoreStage
  );
  let builders = [
    AttrBuilder<(ins "DispatchLoweringPassPipeline":$passPipeline,
        CArg<"unsigned", "0">:$softwarePipelineDepth,
        CArg<"unsigned", "1">:$softwarePipelineStoreStage)>
  ];
  let extraClassDeclaration = [{
    // Returns the lowering pass pipeline set.
    DispatchLoweringPassPipeline getDispatchLoweringPassPipeline();
  }];
  let genVerifyDecl = 1;
}

def IREECodegen_LoweringConfigAttr :
    AttrDef<IREECodegen_Dialect, "LoweringConfig", []> {
  let mnemonic = "lowering_config";
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
    `<` `tile_sizes` `=` $tileSizes
      (`,` `tile_interchange` `=` $tileInterchange^)?
      (`,` `native_vector_size` `=` $nativeVectorSize^)? `>`
  }];

  let parameters = (ins
    AttrParameter<"ArrayAttr",
        "The tile sizes to use for different levels of tiling">:$tileSizes,
    DefaultValuedParameter<"ArrayAttr", "ArrayAttr::get($_ctxt, {})",
        "The tile interchange to use for different levels of tiling">:$tileInterchange,
    DefaultValuedParameter<"ArrayAttr", "ArrayAttr::get($_ctxt, {})",
        "The native vector size to use for the given operation">:$nativeVectorSize
  );
  let builders = [
    AttrBuilder<(ins "TileSizesListTypeRef":$tileSizes,
        CArg<"TileSizesListTypeRef", "{}">:$tileInterchange,
        CArg<"ArrayRef<int64_t>", "{}">:$nativeVectorSize)>
  ];
  let extraClassDeclaration = [{
    // Returns the tile sizes for all levels set for the op.
    TileSizesListType getTileSizeVals();

    // Returns the tile sizes for a level set for the op.
    SmallVector<int64_t> getTileSizeVals(unsigned level);

    // Returns the tile interchange for a level set for the op.
    SmallVector<int64_t> getTileInterchangeVals(unsigned level);

    // Returns the native vector size to use.
    SmallVector<int64_t> getNativeVectorSizeVals();
  }];
  let genVerifyDecl = 1;
}

def IREECodegen_CompilationInfoAttr :
    AttrDef<IREECodegen_Dialect, "CompilationInfo", []> {
  let mnemonic = "compilation_info";
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
    `LoweringConfigAttr` value in the `TranslationInfoAttr` are the
    same since the first-level of tile + distribute is already done
    at the `Flow` level. This verification is also a TODO.
  }];
  let parameters = (ins
    AttrParameter<"LoweringConfigAttr", "">:$loweringConfig,
    AttrParameter<"TranslationInfoAttr", "">:$translationInfo,
    DefaultValuedParameter<"ArrayAttr", "ArrayAttr::get($_ctxt, {})",
        "The workgroup size to use during translation.">:$workgroupSize,
    OptionalParameter<"llvm::Optional<int64_t>",
        "The subgroup size to use during translation.">:$subgroupSize
  );

  let assemblyFormat = [{
    `<` `lowering_config` `=` $loweringConfig `,` `translation_info` `=` $translationInfo
    (`,` `workgroup_size` `=` $workgroupSize^)?
    (`,` `subgroup_size` `=` $subgroupSize^)? `>`
  }];

  // The builder is externally for auto-tuner to generate the attributes.
  let builders = [
    AttrBuilder<(ins "LoweringConfigAttr":$configAttr,
      "TranslationInfoAttr":$translationInfo,
      "ArrayRef<int64_t>":$workgroupSize,
      "llvm::Optional<int64_t>":$subgroupSize
      )>,
  ];
  let extraClassDeclaration = [{
    SmallVector<int64_t> getWorkgroupSizeVals();
  }];
  let genVerifyDecl = 1;
}

#endif // IREE_COMPILER_CODEGEN_DIALECT_LOWERINGCONFIG

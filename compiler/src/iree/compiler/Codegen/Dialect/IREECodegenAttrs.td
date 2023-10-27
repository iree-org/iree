// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_CODEGEN_DIALECT_IREECODEGENATTRS
#define IREE_COMPILER_CODEGEN_DIALECT_IREECODEGENATTRS

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
def CPU_DataTiling
    : I32EnumAttrCase<"CPUDataTiling", 7>;

def LLVMGPU_Default
    : I32EnumAttrCase<"LLVMGPUDefault", 100>;
def LLVMGPU_SimpleDistribute
    : I32EnumAttrCase<"LLVMGPUDistribute", 101>;
def LLVMGPU_Vectorize
    : I32EnumAttrCase<"LLVMGPUVectorize", 102>;
def LLVMGPU_MatmulSimt
    : I32EnumAttrCase<"LLVMGPUMatmulSimt", 103>;
def LLVMGPU_MatmulTensorCore
    : I32EnumAttrCase<"LLVMGPUMatmulTensorCore", 104>;
def LLVMGPU_TransposeSharedMem
    : I32EnumAttrCase<"LLVMGPUTransposeSharedMem", 105>;
def LLVMGPU_WarpReduction
    : I32EnumAttrCase<"LLVMGPUWarpReduction", 106>;
def LLVMGPU_PackUnPack
    : I32EnumAttrCase<"LLVMGPUPackUnPack", 107>;
def LLVMGPU_MatmulTensorCoreMmaSync
    : I32EnumAttrCase<"LLVMGPUMatmulTensorCoreMmaSync", 108>;

def SPIRV_BaseLowering
    : I32EnumAttrCase<"SPIRVBaseLowering", 200>;
def SPIRV_BaseDistribute
    : I32EnumAttrCase<"SPIRVBaseDistribute", 201>;
def SPIRV_BaseVectorize
    : I32EnumAttrCase<"SPIRVBaseVectorize", 202>;
def SPIRV_SubgroupReduce
    : I32EnumAttrCase<"SPIRVSubgroupReduce", 203>;
def SPIRV_MatmulPromoteVectorize
    : I32EnumAttrCase<"SPIRVMatmulPromoteVectorize", 204>;
def SPIRV_CooperativeMatrixVectorize
    : I32EnumAttrCase<"SPIRVCooperativeMatrixVectorize", 205>;
def SPIRV_WinogradVectorize
    : I32EnumAttrCase<"SPIRVWinogradVectorize", 206>;

def VMVX_Default : I32EnumAttrCase<"VMVXDefault", 300>;


def Linalg_TransformDialectCodegen
    : I32EnumAttrCase<"TransformDialectCodegen", 1000>;

def None : I32EnumAttrCase<"None", 0xffff>;

// EnumAttrCase for all known lowerings for ops within dispatch region
// to scalar/native-vector code.
def DispatchLoweringPassPipelineEnum : I32EnumAttr<
  "DispatchLoweringPassPipeline",
  "identifier for pass pipeline use to lower dispatch region", [
    // CPU CodeGen pipelines
    CPU_Default, CPU_DoubleTilingExpert, CPU_DoubleTilingPadExpert,
    CPU_DoubleTilingPeelingExpert, CPU_ConvTileAndDecomposeExpert,
    CPU_Mmt4dTilingExpert, CPU_BufferOpsTileAndVectorize,
    CPU_DataTiling,

    // LLVMGPU CodeGen pipelines
    LLVMGPU_Default, LLVMGPU_SimpleDistribute, LLVMGPU_Vectorize,
    LLVMGPU_MatmulSimt, LLVMGPU_MatmulTensorCore,
    LLVMGPU_TransposeSharedMem, LLVMGPU_WarpReduction,
    LLVMGPU_PackUnPack, LLVMGPU_MatmulTensorCoreMmaSync,

    // SPIR-V CodeGen pipelines
    SPIRV_BaseLowering, SPIRV_BaseDistribute, SPIRV_BaseVectorize,
    SPIRV_SubgroupReduce, SPIRV_MatmulPromoteVectorize,
    SPIRV_CooperativeMatrixVectorize, SPIRV_WinogradVectorize,

    VMVX_Default,

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
    (`store_stage` `=` $softwarePipelineStoreStage^)?
    (`codegen_spec` `=` $codegenSpec^)? `>`
  }];

  let parameters = (ins
    AttrParameter<"IREE::Codegen::DispatchLoweringPassPipelineAttr",
        "Name of the pipeline to be invoked on the translation unit.">:$passPipeline,
    OptionalParameter<"unsigned",
        "The software pipeline depth to be used">:$softwarePipelineDepth,
    DefaultValuedParameter<"unsigned", "1",
        "The software pipeline stage to place stores">:$softwarePipelineStoreStage,
    OptionalParameter<"SymbolRefAttr",
        "The symbol pointing to the transform dialect codegen spec to be used">:$codegenSpec
  );
  let builders = [
    AttrBuilder<(ins "DispatchLoweringPassPipeline":$passPipeline,
        CArg<"unsigned", "0">:$softwarePipelineDepth,
        CArg<"unsigned", "1">:$softwarePipelineStoreStage,
        CArg<"SymbolRefAttr", "{}">:$codegenSpec)>
  ];
  let extraClassDeclaration = [{
    // Returns the lowering pass pipeline set.
    DispatchLoweringPassPipeline getDispatchLoweringPassPipeline();
  }];
  let genVerifyDecl = 1;
}

def IREECodegen_LoweringConfigTilingLevelAttr :
  AttrDef<IREECodegen_Dialect, "LoweringConfigTilingLevel", []>
{
  let mnemonic = "lowering_config_level";
  let parameters = (ins
    ArrayRefParameter<"int64_t",
        "The tile sizes to use for this level of tiling">:$sizes,
    OptionalArrayRefParameter<"int64_t",
        "The tile interchange to use for this level of tiling">:$interchange,
    OptionalArrayRefParameter<"bool",
        "The scalable tile flags for this level of tiling">:$scalableFlags);
  let hasCustomAssemblyFormat = 1;
  let genVerifyDecl = 1;
}

def IREECodegen_LoweringConfigTilingLevelsAttr :
  ArrayOfAttr<IREECodegen_Dialect, "LoweringConfigTilingLevels",
    "lowering_config_levels", "LoweringConfigTilingLevelAttr", []>
{
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
    `<` `tile_sizes` `=` $tilingLevels
      (`,` `native_vector_size` `=` `[` $nativeVectorSize^ `]`)? `>`
  }];

  let parameters = (ins
    AttrParameter<"LoweringConfigTilingLevelsAttr",
        "The lowering config at different levels">:$tilingLevels,
    OptionalArrayRefParameter<"int64_t",
        "The native vector size to use for the given operation">:$nativeVectorSize
  );
  let builders = [
    AttrBuilder<(ins "TileSizesListTypeRef":$tileSizes,
        CArg<"TileSizesListTypeRef", "{}">:$tileInterchange,
        CArg<"ArrayRef<int64_t>", "{}">:$nativeVectorSize)>,
    AttrBuilder<(ins "TileSizesListTypeRef":$tileSizes,
        "ScalableTileFlagsListTypeRef":$scalableTileFlags,
        CArg<"TileSizesListTypeRef", "{}">:$tileInterchange,
        CArg<"ArrayRef<int64_t>", "{}">:$nativeVectorSize)>
  ];
  let extraClassDeclaration = [{
    // Returns the tile sizes for all levels set for the op.
    TileSizesListType getTileSizeVals();

    // Returns the tile sizes for a level set for the op.
    SmallVector<int64_t> getTileSizeVals(unsigned level);

    // Returns the scalable tile flags for all levels set for the op.
    ScalableTileFlagsListType getScalableTileFlagVals();

    // Returns the scalable tile flags for a level set for the op.
    SmallVector<bool> getScalableTileFlagVals(unsigned level);

    // Returns the tile interchange for a level set for the op.
    SmallVector<int64_t> getTileInterchangeVals(unsigned level);

    // Returns the native vector size to use.
    SmallVector<int64_t> getNativeVectorSizeVals() {
      return SmallVector<int64_t>(getNativeVectorSize());
    }
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
    OptionalArrayRefParameter<"int64_t", "The workgroup size to use during translation.">:$workgroupSize,
    OptionalParameter<"std::optional<int64_t>",
        "The subgroup size to use during translation.">:$subgroupSize
  );

  let assemblyFormat = [{
    `<` `lowering_config` `=` $loweringConfig `,` `translation_info` `=` $translationInfo
    (`,` `workgroup_size` `=` `[` $workgroupSize^ `]`)?
    (`,` `subgroup_size` `=` $subgroupSize^)? `>`
  }];

  let extraClassDeclaration = [{
    SmallVector<int64_t> getWorkgroupSizeVals() {
      return SmallVector<int64_t>(getWorkgroupSize());
    }
  }];
  let genVerifyDecl = 1;
}

def IREECodegen_ExportConfig : AttrDef<IREECodegen_Dialect, "ExportConfig", []> {
  let mnemonic = "export_config";
  let summary = "User defined workgroup size specification";
  let description = [{
    Allows setting workgroup size for pre-formed dispatches.
  }];
  let parameters = (ins
    ArrayRefParameter<"int64_t", "Workgroup Size to use">:$workgroup_size
  );
  let assemblyFormat = [{
    `<` `workgroup_size` `=` `[` $workgroup_size `]` `>`
  }];
  let extraClassDeclaration = [{
    ArrayAttr getWorkgroupSizeIndexArray();
  }];
  let genVerifyDecl = 1;
}

#endif // IREE_COMPILER_CODEGEN_DIALECT_IREECODEGENATTRS

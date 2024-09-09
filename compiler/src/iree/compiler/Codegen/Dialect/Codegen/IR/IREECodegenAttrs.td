// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_CODEGEN_DIALECT_IREECODEGENATTRS
#define IREE_COMPILER_CODEGEN_DIALECT_IREECODEGENATTRS

include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenDialect.td"
include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenInterfaces.td"
include "mlir/Dialect/SCF/IR/DeviceMappingInterface.td"
include "mlir/IR/EnumAttr.td"

//===---------------------------------------------------------------------===//
// Pass pipelines
//===---------------------------------------------------------------------===//

// List of pre-existing pipelines for translating executables.
def CPU_Default
    : I32EnumAttrCase<"CPUDefault", 0>;
def CPU_DoubleTilingExpert
    : I32EnumAttrCase<"CPUDoubleTilingExpert", 1>;
def CPU_ConvTileAndDecomposeExpert
    : I32EnumAttrCase<"CPUConvTileAndDecomposeExpert", 2>;
def CPU_Mmt4dTilingExpert
    : I32EnumAttrCase<"Mmt4dTilingExpert", 3>;
def CPU_BufferOpsTileAndVectorize
    : I32EnumAttrCase<"CPUBufferOpsTileAndVectorize", 4>;
def CPU_DataTiling
    : I32EnumAttrCase<"CPUDataTiling", 5>;
def CPU_LinalgExtTileAndVectorize
    : I32EnumAttrCase<"CPULinalgExtTileAndVectorize", 6>;

def LLVMGPU_Default
    : I32EnumAttrCase<"LLVMGPUDefault", 100>;
def LLVMGPU_BaseLowering
    : I32EnumAttrCase<"LLVMGPUBaseLowering", 101>;
def LLVMGPU_SimpleDistribute
    : I32EnumAttrCase<"LLVMGPUDistribute", 102>;
def LLVMGPU_Vectorize
    : I32EnumAttrCase<"LLVMGPUVectorize", 103>;
def LLVMGPU_MatmulSimt
    : I32EnumAttrCase<"LLVMGPUMatmulSimt", 104>;
def LLVMGPU_MatmulTensorCore
    : I32EnumAttrCase<"LLVMGPUMatmulTensorCore", 105>;
def LLVMGPU_TransposeSharedMem
    : I32EnumAttrCase<"LLVMGPUTransposeSharedMem", 106>;
def LLVMGPU_WarpReduction
    : I32EnumAttrCase<"LLVMGPUWarpReduction", 107>;
def LLVMGPU_PackUnPack
    : I32EnumAttrCase<"LLVMGPUPackUnPack", 108>;
def LLVMGPU_MatmulTensorCoreMmaSync
    : I32EnumAttrCase<"LLVMGPUMatmulTensorCoreMmaSync", 109>;
def LLVMGPU_VectorDistribute
    : I32EnumAttrCase<"LLVMGPUVectorDistribute", 110>;
def LLVMGPU_PadAndVectorDistribute
    : I32EnumAttrCase<"LLVMGPUPadAndVectorDistribute", 111>;
def LLVMGPU_WinogradVectorize
    : I32EnumAttrCase<"LLVMGPUWinogradVectorize", 112>;
def LLVMGPU_TileAndFuse
    : I32EnumAttrCase<"LLVMGPUTileAndFuse", 113>;

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
def Custom
    : I32EnumAttrCase<"Custom", 1001>;

def None : I32EnumAttrCase<"None", 0xffff>;

// EnumAttrCase for all known lowerings for ops within dispatch region
// to scalar/native-vector code.
def DispatchLoweringPassPipelineEnum : I32EnumAttr<
  "DispatchLoweringPassPipeline",
  "identifier for pass pipeline use to lower dispatch region", [
    // CPU CodeGen pipelines
    CPU_Default, CPU_DoubleTilingExpert,
    CPU_ConvTileAndDecomposeExpert,
    CPU_Mmt4dTilingExpert, CPU_BufferOpsTileAndVectorize,
    CPU_DataTiling, CPU_LinalgExtTileAndVectorize,

    // LLVMGPU CodeGen pipelines
    LLVMGPU_Default, LLVMGPU_BaseLowering, LLVMGPU_SimpleDistribute,
    LLVMGPU_Vectorize, LLVMGPU_MatmulSimt, LLVMGPU_MatmulTensorCore,
    LLVMGPU_TransposeSharedMem, LLVMGPU_WarpReduction, LLVMGPU_PackUnPack,
    LLVMGPU_MatmulTensorCoreMmaSync, LLVMGPU_VectorDistribute,
    LLVMGPU_PadAndVectorDistribute, LLVMGPU_WinogradVectorize,
    LLVMGPU_TileAndFuse,

    // SPIR-V CodeGen pipelines
    SPIRV_BaseLowering, SPIRV_BaseDistribute, SPIRV_BaseVectorize,
    SPIRV_SubgroupReduce, SPIRV_MatmulPromoteVectorize,
    SPIRV_CooperativeMatrixVectorize, SPIRV_WinogradVectorize,

    VMVX_Default,

    // Transform dialect based codegen
    Linalg_TransformDialectCodegen,

    // For out of tree pass-pipelines
    Custom,

    // None to specify no in-built pipelines to use.
    None
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

//===---------------------------------------------------------------------===//
// IREE Codegen workgroup mapping attributes
//===---------------------------------------------------------------------===//

def IdX : I64EnumAttrCase<"IdX", 0, "x">;
def IdY : I64EnumAttrCase<"IdY", 1, "y">;
def IdZ : I64EnumAttrCase<"IdZ", 2, "z">;

def WorkgroupIdEnum :
    I64EnumAttr<"WorkgroupId", "Attribute that map to hal.workgrpoup.ids", [
      IdX, IdY, IdZ]> {
  let cppNamespace = "::mlir::iree_compiler::IREE::Codegen";
}

def WorkgroupMappingAttr :
    AttrDef<IREECodegen_Dialect, "WorkgroupMapping", [
      DeclareAttrInterfaceMethods<DeviceMappingAttrInterface>]> {
  let mnemonic = "workgroup_mapping";

  let parameters = (ins
    EnumParameter<WorkgroupIdEnum>:$id,
    DefaultValuedParameter<"int64_t", "0">:$delinearizedDim
  );
  let assemblyFormat = "`<` $id (`` `:` `` $delinearizedDim^)? `>`";

  let description = [{
    Attribute that eventually will be used to map distributed loop iterations
    to `hal.workgroup.id`s.

    The `x`,`y` and `z` values for `id` map to `hal.workgroup.id[0]`,
    `hal.workgroup.id[1]` and `hal.workgroup.id[2]` respectively.

    In addition it is possible to specify if the `z` dimension is to be
    delinearized on mapping. For example if the list of mapping attributes is
    `[workgroup_mapping<z:1>, workgroup_mapping<z:0>]`, then the `z` dimension
    is delinearized to map to `workgroup_mapping<z:1>` and
    `workgroup_mapping<z:0>`. In other words if the number of logical parallel
    workers along the `z:0` dimension is `W`, then
    ```
    workgroup_mapping<z:0> = hal.workgroup.id[1] mod W,
    worgrkoup_mapping<z:1> = hal.workgroup.id[1] div W
    ```

    Note: It is expected that this attribute is always used in a list of mapping
    attributes (with a single element being a list of size 1). It is illegal for
    a list to have `workgroup_mapping<z:a>` without `workgroup_mapping<z:b>`
    if `a > b`. In the same way it is illegal to for the list to
    - have `workgroup_mapping<y>` but not `workgroup_mapping<x>`
    - have `workgroup_mapping<z:*>`  but not have `workgroup_mapping<x>`
      and `workgroup_mapping<y>`
  }];
  let builders = [
    AttrBuilder<(ins "WorkgroupId":$id)>
  ];
  let extraClassDeclaration = [{
    // Checks that a list of attributes is well-defined.
    static LogicalResult verifyAttrList(::mlir::MLIRContext *context,
      Location loc, ArrayRef<Attribute> attrs);

    // Convert from mapping ID to attribute.
    static ::mlir::iree_compiler::IREE::Codegen::WorkgroupMappingAttr
    getAttributeFromMappingId(MLIRContext *context, int64_t mappingId);

    // Less than operator for easy comparison.
    bool operator<(
        const ::mlir::iree_compiler::IREE::Codegen::WorkgroupMappingAttr &rhs)
        const;
  }];

  let genVerifyDecl = 1;
}

//===---------------------------------------------------------------------===//
// iree_codegen.translation_info
//===---------------------------------------------------------------------===//

def IREECodegen_TranslationInfoAttr :
    AttrDef<IREECodegen_Dialect, "TranslationInfo", []> {
  let mnemonic = "translation_info";
  let summary = [{drive dispatch entry point lowering}];
  let description = [{
    Specifies the information that is used to drive the translation of
    an entry point function using Linalg based structured-op
    lowering. During executable translation this is attached to the
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
    (`codegen_spec` `=` $codegenSpec^)?
    (`workgroup_size` `=` `[` $workgroupSize^ `]`)?
    (`subgroup_size` `=` $subgroupSize^)?
    (`,` $configuration^)? `>`
  }];

  let parameters = (ins
    AttrParameter<"IREE::Codegen::DispatchLoweringPassPipelineAttr",
        "Name of the pipeline to be invoked on the translation unit.">:$passPipeline,
    OptionalParameter<"SymbolRefAttr",
        "The symbol pointing to the transform dialect codegen spec to be used">:$codegenSpec,
    OptionalArrayRefParameter<"int64_t", "The workgroup size to use">:$workgroupSize,
    OptionalParameter<"int64_t", "The subgroup size to use">:$subgroupSize,
    OptionalParameter<"DictionaryAttr",
        "Pipeline specific configuration">:$configuration
  );
  let builders = [
    AttrBuilder<(ins "DispatchLoweringPassPipeline":$passPipeline,
        CArg<"SymbolRefAttr", "{}">:$codegenSpec,
        CArg<"ArrayRef<int64_t>", "{}">:$workgroupSize,
        CArg<"std::optional<int64_t>", "std::nullopt">:$subgroupSize,
        CArg<"DictionaryAttr", "{}">:$configuration)>,
    AttrBuilder<(ins "DispatchLoweringPassPipeline":$passPipeline,
        "ArrayRef<int64_t>":$workgroupSize,
        CArg<"std::optional<int64_t>", "std::nullopt">:$subgroupSize,
        CArg<"DictionaryAttr", "{}">:$configuration)>
  ];
  let extraClassDeclaration = [{
    // Returns the lowering pass pipeline set.
    DispatchLoweringPassPipeline getDispatchLoweringPassPipeline();
  }];
  let genVerifyDecl = 1;
}

//===---------------------------------------------------------------------===//
// iree_codegen.lowering_config
//===---------------------------------------------------------------------===//

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
    AttrDef<IREECodegen_Dialect, "LoweringConfig", [
      DeclareAttrInterfaceMethods<IREECodegen_LoweringConfigAttrInterface, [
        "getWorkgroupTileSizes",
        "getWorkgroupInterchange",
        "getStaticTilingLevelSizes",
        "getTilingLevelSizes",
        "hasTilingLevel",
      ]>
    ]> {
  let mnemonic = "lowering_config";
  let summary = [{drive lowering of an operation within dispatch region}];
  let description = [{
    Default implementation of a lowering configuration attribute. It includes
    only tiling and optionally vectorization information. The interpretation of
    the tiles sizes are backend dependent.

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
    TileSizesListType getTileSizeVals() const;

    // Returns the tile sizes for a level set for the op.
    SmallVector<int64_t> getTileSizeVals(unsigned level) const;

    // Returns the scalable tile flags for all levels set for the op.
    ScalableTileFlagsListType getScalableTileFlagVals();

    // Returns the scalable tile flags for a level set for the op.
    SmallVector<bool> getScalableTileFlagVals(unsigned level);

    // Returns the tile interchange for a level set for the op.
    SmallVector<int64_t> getTileInterchangeVals(unsigned level) const;

    // Returns true if there are no tile interchange values (this means that
    // interchange can be ignored).
    bool isInterchangeEmpty();

    // Returns the native vector size to use.
    SmallVector<int64_t> getNativeVectorSizeVals() {
      return SmallVector<int64_t>(getNativeVectorSize());
    }
  }];

  let genVerifyDecl = 1;
}

//===---------------------------------------------------------------------===//
// iree_codegen.compilation_info
//===---------------------------------------------------------------------===//

def IREECodegen_CompilationInfoAttr :
    AttrDef<IREECodegen_Dialect, "CompilationInfo", []> {
  let mnemonic = "compilation_info";
  let summary = [{drive lowering of an operation from input dialect}];
  let description = [{
    Specifies the information that allows controlling the compilation
    of operations like `linalg.matmul`/`linalg.*conv` within
    IREE. This information is used to override the defaults used by
    the IREE compiler. If set on the input to the compiler, there is no
    guarantee that the config survives until codegen. Named operations like
    `linalg.matmul`/`linalg.*conv*` are more likely to retain their lowering
    configurations.

    TODO: It is expected that the `TranslationInfoAttr` and the
    `LoweringConfigAttr` are specified. Currently there is no
    verification that the values of the `LoweringConfigAttr` fully
    specifies the behaviour of the compilation path chosen with
    `TranslationInfoAttr`. This could be added in the future.
  }];
  let parameters = (ins
    AttrParameter<"LoweringConfigAttrInterface", "">:$loweringConfig,
    AttrParameter<"TranslationInfoAttr", "">:$translationInfo
  );

  let assemblyFormat = [{
    `<` `lowering_config` `=` $loweringConfig `,` `translation_info` `=` $translationInfo `>`
  }];

  let genVerifyDecl = 1;
}

//===---------------------------------------------------------------------===//
// iree_codegen.export_config
//===---------------------------------------------------------------------===//

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

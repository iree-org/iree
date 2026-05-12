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
// IREE Codegen local mapping attribute
//===---------------------------------------------------------------------===//

def LocalMappingAttr :
    AttrDef<IREECodegen_Dialect, "LocalMapping", [
      DeclareAttrInterfaceMethods<DeviceMappingAttrInterface>]> {
  let mnemonic = "local_mapping";

  let parameters = (ins
    "int64_t":$dim
  );
  let assemblyFormat = "`<` $dim `>`";

  let description = [{
    Attribute for mapping `scf.forall` loops to sequential local execution.

    Example:
    ```mlir
      scf.forall (%i, %j) in (4, 8) {
        // body ... %i, %j
      } {mapping = [#iree_codegen.local_mapping<1>,
                    #iree_codegen.local_mapping<0>]}
    ```

    This forall approximately corresponds to the following loop nest:

    ```mlir
      scf.for %i = 0 to 4 {
        scf.for %j = 0 to 8 {
        // body ... %i, %j
        }
      }
    ```
  }];
}

//===---------------------------------------------------------------------===//
// iree_codegen.simple_target
//===---------------------------------------------------------------------===//

def IREECodegen_SimpleTargetAttr :
  AttrDef<IREECodegen_Dialect, "SimpleTarget", [
    DeclareAttrInterfaceMethods<IREECodegen_TargetInfoAttrInterface, [
      "getMaximumWorkgroupCount"]>]> {
  let summary = "Default implementation of TargetInfoAttrInterface";

  let description = [{
    Implements the basic information needed to satisfy the
    `TargetInfoAttrInterface`
  }];

  let mnemonic = "simple_target";
  let parameters = (ins
    OptionalArrayRefParameter<"int64_t",
      "Maximum allowed workgroup count in [x, y, z]">:$max_workgroup_count
  );

  let assemblyFormat = [{`<`
    (`max_workgroup_count` `=` `[` $max_workgroup_count^ `]` )?
  `>`}];

  let genVerifyDecl = 1;
}


//===---------------------------------------------------------------------===//
// iree_codegen pipeline attributes
//===---------------------------------------------------------------------===//

class IREECodegen_UnitPipelineAttr<string name, string mnemonicStr,
                                   string summaryStr> :
    AttrDef<IREECodegen_Dialect, name, [
      DeclareAttrInterfaceMethods<IREECodegen_PipelineAttrInterface, [
        "buildPipeline"
      ]>
    ]> {
  let mnemonic = mnemonicStr;
  let summary = summaryStr;
  let parameters = (ins);
  let assemblyFormat = [{}];
}

def IREECodegen_VMVXPipelineAttr :
    IREECodegen_UnitPipelineAttr<"VMVXPipeline", "vmvx_pipeline",
                                 "The VMVX lowering pipeline.">;

def IREECodegen_TransformDialectCodegenPipelineAttr :
    IREECodegen_UnitPipelineAttr<
        "TransformDialectCodegenPipeline", "transform_dialect_codegen",
        "Transform dialect driven codegen pipeline marker.">;

def IREECodegen_NoPipelineAttr :
    IREECodegen_UnitPipelineAttr<"NoPipeline", "no_pipeline",
                                 "No-op codegen pipeline marker.">;

def IREECodegen_PassPipelineAttr :
    AttrDef<IREECodegen_Dialect, "PassPipeline", [
      DeclareAttrInterfaceMethods<IREECodegen_PipelineAttrInterface, [
        "buildPipeline"
      ]>
    ]> {
  let mnemonic = "pass_pipeline";
  let summary = "An attribute carrying a textual pass pipeline string.";
  let description = [{
    Specifies a pass pipeline using MLIR's textual pass pipeline syntax.
    The pipeline string is parsed and populated into an OpPassManager
    when `buildPipeline` is called.
  }];
  let parameters = (ins
    StringRefParameter<"The textual pass pipeline specification">:$pipeline
  );
  let assemblyFormat = [{
    `<` $pipeline `>`
  }];
  let genVerifyDecl = 1;
}

//===---------------------------------------------------------------------===//
// iree_codegen.translation_info
//===---------------------------------------------------------------------===//

def IREECodegen_TranslationInfoAttr :
    AttrDef<IREECodegen_Dialect, "TranslationInfo", []> {
  let mnemonic = "translation_info";
  let summary = [{Drive dispatch entry point lowering.}];
  let description = [{
    Specifies the information that is used to drive the translation of
    an entry point function using Linalg based structured-op
    lowering. During executable translation this is attached to the
    `hal.executable.export` operation.

    If this operation is already set on the root operation (as part of
    `iree_codegen.compilation_info`) that drives the compilation of a
    dispatch region (like `linalg.matmul`/`linalg.*conv*`), this
    attribute gets propagated to the entry point function.

    The `passPipeline` field must be an attribute implementing
    `PipelineAttrInterface` (e.g. `#iree_codegen.no_pipeline` or
    `#iree_codegen.pass_pipeline<"...">`).
  }];

  let parameters = (ins
    AttrParameter<"Attribute",
        "Pass pipeline specification.">:$passPipeline,
    OptionalParameter<"SymbolRefAttr",
        "The symbol pointing to the transform dialect codegen spec to be used">:$codegenSpec,
    OptionalArrayRefParameter<"int64_t", "The workgroup size to use">:$workgroupSize,
    OptionalParameter<"int64_t", "The subgroup size to use">:$subgroupSize,
    OptionalParameter<"DictionaryAttr",
        "Pipeline specific configuration">:$configuration
  );

  let assemblyFormat = [{
    `<` `pipeline` `=` $passPipeline
      (`codegen_spec` `=` $codegenSpec^)?
      (`workgroup_size` `=` `[` $workgroupSize^ `]`)?
      (`subgroup_size` `=` $subgroupSize^)?
      (`,` $configuration^)?
    `>`
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
        "getNumTilingLevels",
        "getStaticTilingLevelSizes",
        "getTilingLevelAttr",
        "getTilingLevelSizes",
        "hasTilingLevel",
        "hasWorkgroupTilingLevel",
      ]>
    ]> {
  let mnemonic = "lowering_config";
  let summary = [{Drive lowering of an operation within dispatch region.}];
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
    `<` `tile_sizes` `=` $tilingLevels `>`
  }];

  let parameters = (ins
    AttrParameter<"LoweringConfigTilingLevelsAttr",
        "The lowering config at different levels">:$tilingLevels
  );
  let builders = [
    AttrBuilder<(ins "TileSizesListTypeRef":$tileSizes,
        CArg<"TileSizesListTypeRef", "{}">:$tileInterchange)>,
    AttrBuilder<(ins "TileSizesListTypeRef":$tileSizes,
        "ScalableTileFlagsListTypeRef":$scalableTileFlags,
        CArg<"TileSizesListTypeRef", "{}">:$tileInterchange)>
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
  }];

  let genVerifyDecl = 1;
}

//===---------------------------------------------------------------------===//
// iree_codegen.compilation_info
//===---------------------------------------------------------------------===//

def IREECodegen_CompilationInfoAttr :
    AttrDef<IREECodegen_Dialect, "CompilationInfo", []> {
  let mnemonic = "compilation_info";
  let summary = [{Drive lowering of an operation from input dialect.}];
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
  let summary = [{User defined workgroup size specification.}];
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

//===---------------------------------------------------------------------===//
// iree_codegen.rotate_rows
//===---------------------------------------------------------------------===//

def IREECodegen_RotateRowsAttr  :
    AttrDef<IREECodegen_Dialect, "RotateRows", [
    DeclareAttrInterfaceMethods<IREECodegen_SwizzleAttrInterface, [
        "swizzleOffset",
        "getAccessElementCount"
      ]>
    ]> {
  let mnemonic = "rotate_rows";
  let summary = [{An attribute that describes a swizzling pattern for rotating rows.}];
  let description = [{
    This attribute rotates accesses of |access_width| within rows of size
    |row_width|. For any given access into logical memref of shape
    `memref<...xNx|access_width|x!eltype>` where `N = row_width / access_width`
    at position `(i, j, 0)` is rotated to `(i, (i + j) % N, 0)`. For example,

    ```
    row_width = 16, access_width = 4

    0000 1111 2222 3333 /// 0 1 2 3
    4444 5555 6666 7777 /// 0 1 2 3
    8888 9999 AAAA BBBB /// 0 1 2 3
    CCCC DDDD EEEE FFFF /// 0 1 2 3
    ```

    is swizzled to
    ```
    0000 1111 2222 3333 /// 0 1 2 3
    7777 4444 5555 6666 /// 3 0 1 2
    AAAA BBBB 8888 9999 /// 2 3 0 1
    DDDD EEEE FFFF CCCC /// 1 2 3 0
    ```

    The pattern repeats for subsequent rows.
  }];
  let parameters = (ins
    AttrParameter<"int64_t", "">:$row_width,
    AttrParameter<"int64_t", "">:$access_width
  );
  let assemblyFormat = [{
    `<` $row_width `,` $access_width `>`
  }];
  let extraClassDeclaration = [{
    int64_t getSwizzlePeriod() const {
      return getRowWidth() * (getRowWidth() / getAccessWidth());
    }
  }];
  let genVerifyDecl = 1;
}

//===---------------------------------------------------------------------===//
// iree_codegen.symbolic_ukernel_provider
//===---------------------------------------------------------------------===//

def IREECodegen_SymbolicUKernelProviderAttr  :
    AttrDef<IREECodegen_Dialect, "SymbolicUKernelProvider", [
    DeclareAttrInterfaceMethods<IREECodegen_UKernelProviderInterface, [
        "getMLIRUKernel"
      ]>
    ]> {
  let mnemonic = "symbolic_ukernel_provider";
  let summary = [{
    An attribute that provides ukernel implementations by looking up the
    nearest symbol table from the annotation site.
  }];
  let parameters = (ins);
  let assemblyFormat = [{}];
}

//===---------------------------------------------------------------------===//
// iree_codegen.ukernel_descriptor
//===---------------------------------------------------------------------===//

def UKernelKind_Tensor : I64EnumAttrCase<"Tensor", 0, "tensor">;
def UKernelKind_Memref : I64EnumAttrCase<"Memref", 1, "memref">;
def UKernelKind_Bitcode : I64EnumAttrCase<"Bitcode", 2, "bitcode">;

def UKernelArgumentKindEnum :
    I64EnumAttr<"UKernelArgumentKind", "Attribute describing the ukernel integration point",
    [
      UKernelKind_Tensor,
      UKernelKind_Memref,
      UKernelKind_Bitcode
    ]> {
  let cppNamespace = "::mlir::iree_compiler::IREE::Codegen";
}

def IREECodegen_UKernelDescriptorAttr  :
    AttrDef<IREECodegen_Dialect, "UKernelDescriptor"> {
  let mnemonic = "ukernel_descriptor";
  let summary = [{
    An attribute that specifies the ukernel implementation based on the |ukernel_name| and
    the integration point based on the |kind|. Example integration points could for example be
    at the tensor, memref or bitcode level.
  }];
  let parameters = (ins
    StringRefParameter<>:$ukernel_name,
    EnumParameter<UKernelArgumentKindEnum>:$kind
  );
  let assemblyFormat = [{
    `<` $ukernel_name `,` $kind `>`
  }];
}


//===---------------------------------------------------------------------===//
// iree_codegen.xor_shuffle
//===---------------------------------------------------------------------===//

def IREECodegen_XORShuffleAttr  :
    AttrDef<IREECodegen_Dialect, "XORShuffle", [
    DeclareAttrInterfaceMethods<IREECodegen_SwizzleAttrInterface, [
        "swizzleOffset",
        "getAccessElementCount"
      ]>
    ]> {
  let mnemonic = "xor_shuffle";
  let summary = "An attribute that describes an XOR-based swizzling pattern.";
  let description = [{
    Shuffles accesses of |access_width| within rows of size
    |row_width|. For any given access into logical memref of shape
    `memref<...xNx|access_width|x!eltype>` where `N = row_width / access_width`
    at position `(i, j, 0)` is shuffled to `(i, ((i/per_phase) %N) XOR j , 0)`. For example,

    ```
    row_width = 16, access_width = 4, per_phase = 1

    0000 1111 2222 3333 /// 0 1 2 3
    4444 5555 6666 7777 /// 0 1 2 3
    8888 9999 AAAA BBBB /// 0 1 2 3
    CCCC DDDD EEEE FFFF /// 0 1 2 3
    ```

    is swizzled to
    ```
    0000 1111 2222 3333 /// 0 1 2 3
    5555 4444 7777 6666 /// 1 0 3 2
    AAAA BBBB 8888 9999 /// 2 3 0 1
    FFFF EEEE DDDD CCCC /// 3 2 1 0
    ```
    |per_phase| allows to keep the same shuffling across multiple rows. For example,

    ```
    row_width = 16, access_width = 4, per_phase = 2

    0000 1111 2222 3333 /// 0 1 2 3
    4444 5555 6666 7777 /// 0 1 2 3
    8888 9999 AAAA BBBB /// 0 1 2 3
    CCCC DDDD EEEE FFFF /// 0 1 2 3
    ```

    is swizzled to
    ```
    0000 1111 2222 3333 /// 0 1 2 3
    4444 5555 6666 7777 /// 0 1 2 3
    9999 8888 BBBB AAAA /// 1 0 3 2
    DDDD CCCC FFFF EEEE /// 1 0 3 2
    ```

    The pattern repeats for subsequent rows.
  }];
  let parameters = (ins
    AttrParameter<"int64_t", "">:$row_width,
    AttrParameter<"int64_t", "">:$access_width,
    OptionalParameter<"int64_t", "row stride. Default to row_width">:$row_stride,
    OptionalParameter<"int64_t", "Default to 1">:$per_phase
  );
  let assemblyFormat = [{
    `<` $row_width `,` $access_width  (`,` $row_stride^)? (`,` $per_phase^)? `>`
  }];
  let extraClassDeclaration = [{
    int64_t getSwizzlePeriod() const {
      return getRowWidth() * (getRowWidth() / getAccessWidth());
    }
  }];
  let genVerifyDecl = 1;
}

//===---------------------------------------------------------------------===//
// iree_codegen.denormal_mode
//===---------------------------------------------------------------------===//

// Do not add any denormal mode.
def DenormalMode_None : I32EnumCase<"None", 0, "none">;
// Convert denormals to zero while preserving sign.
def DenormalMode_PreserveSign : I32EnumCase<"PreserveSign", 1, "preserve-sign">;
// Convert denormals to positive zero.
def DenormalMode_PositiveZero : I32EnumCase<"PositiveZero", 2, "positive-zero">;

// Denormal fp mode.
def DenormalFpMathEnum : I32Enum<
  "DenormalFpMath",
  "Denormal mode for fp math", [
    DenormalMode_None,
    DenormalMode_PreserveSign,
    DenormalMode_PositiveZero
  ]> {
  let cppNamespace = "::mlir::iree_compiler::IREE::Codegen";
}

def DenormalFpMathAttr :
  EnumAttr<IREECodegen_Dialect, DenormalFpMathEnum, "denormal_fp_math"> {
  let assemblyFormat = "`<` $value `>`";

  let extraClassDeclaration = [{
    /// Returns the key name for fp32 DenormalFpMathAttr in a DictionaryAttr.
    static StringRef getFP32DictKeyName() {
      return "iree_codegen.denormal_fp_math_f32";
    }
  }];
}

//===---------------------------------------------------------------------===//
// iree_codegen.workgroup_scope
//===---------------------------------------------------------------------===//

def WorkgroupScopeAttr : AttrDef<IREECodegen_Dialect, "WorkgroupScope", []> {
  let mnemonic = "workgroup_scope";
  let summary = [{
    Attribute representing parallel execution across workgroups.
  }];
  let description = [{
    Parallel scope that maps to workgroup execution.

    If |linearize| is set to true, workgroup ids are linearized into a single
    id which is then typically delinearized based on the problem space.
  }];
  let parameters = (ins "bool":$linearize);
  let hasCustomAssemblyFormat = 1;

  // Interface is registered as an external model to avoid HAL dep.
}

//===---------------------------------------------------------------------===//
// iree_codegen.root_op
//===---------------------------------------------------------------------===//

def IREECodegen_RootOpAttr : AttrDef<IREECodegen_Dialect, "RootOp"> {
  let mnemonic = "root_op";
  let summary = "Marks an operation as a tuner root op in a given set";
  let description = [{
    Attribute identifying a root operation. The `set` parameter groups root ops
    into numbered sets: all root ops in the given set share the same
    `lowering_config`. Codegen does not rely on the `root_op` attribute; it is
    only used for constraint generation when tuning.
  }];
  let parameters = (ins "int64_t":$set);
  let assemblyFormat = "`<` `set` `=` $set `>`";
}

//===---------------------------------------------------------------------===//
// iree_codegen.smt.int_knob
//===---------------------------------------------------------------------===//

def IREECodegen_IntKnobAttr : AttrDef<IREECodegen_Dialect, "IntKnob"> {
  let mnemonic = "smt.int_knob";
  let summary = "Integer-valued tunable knob placeholder.";
  let description = [{
    Represents a named placeholder for an integer tunable parameter in a
    constraints knobs dictionary. During constraint generation, these appear
    in tiling arrays (workgroup, reduction, thread), workgroup_size, and
    subgroup_size positions. The name matches the corresponding
    `iree_codegen.smt.knob` op name.
  }];
  let parameters = (ins "StringAttr":$name);
  let assemblyFormat = "`<` $name `>`";
  let builders = [
    AttrBuilder<(ins "StringRef":$name), [{
      return $_get($_ctxt, StringAttr::get($_ctxt, name));
    }]>
  ];
}

//===---------------------------------------------------------------------===//
// iree_codegen.smt.one_of_knob
//===---------------------------------------------------------------------===//

def IREECodegen_OneOfKnobAttr : AttrDef<IREECodegen_Dialect, "OneOfKnob"> {
  let mnemonic = "smt.one_of_knob";
  let summary = "Select-from-enumerated-set knob placeholder.";
  let description = [{
    Represents a named placeholder for a discrete choice from a fixed set of
    attributes. The SMT variable is an integer index into the options array.
    During extraction, the actual attribute is matched against the options to
    find its index. During materialization, the index selects directly from
    the array -- no reverse lookup needed.
  }];
  let parameters = (ins "StringAttr":$name, "ArrayAttr":$options);
  let assemblyFormat = "`<` $name `,` $options `>`";
}

#endif // IREE_COMPILER_CODEGEN_DIALECT_IREECODEGENATTRS

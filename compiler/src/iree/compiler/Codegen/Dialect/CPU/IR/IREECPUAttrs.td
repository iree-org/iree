// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_CODEGEN_DIALECT_CPU_IREECPUATTRS
#define IREE_COMPILER_CODEGEN_DIALECT_CPU_IREECPUATTRS

include "iree/compiler/Codegen/Dialect/CPU/IR/IREECPUDialect.td"
include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenInterfaces.td"
include "mlir/IR/AttrTypeBase.td"

//===----------------------------------------------------------------------===//
// CPU Specific Lowering Config Attributes
//===----------------------------------------------------------------------===//

def IREECPU_LoweringConfigAttr :
    AttrDef<IREECPU_Dialect, "LoweringConfig", [
      DeclareAttrInterfaceMethods<IREECodegen_LoweringConfigAttrInterface, [
        "getWorkgroupTileSizes",
        "getTilingLevelSizes",
        "hasTilingLevel",
        "hasWorkgroupTilingLevel",
        "getNumTilingLevels",
        "getStaticTilingLevelSizes",
        "getTilingLevelAttr",
        "getVectorSizes",
        "getVectorScalableFlags",
      ]>
    ]> {
  let mnemonic = "lowering_config";
  let summary = [{Drive lowering of an operation for cpu compilation.}];
  let description = [{
    CPU specific implementation of a lowering config. This carries just a
    dictionary attribute to store any relevant fields. This is the simplest
    form of a lowering config, offering flexibility at the cost of structure.

    For some key entries, e.g., distribution, etc., they must be
    IREE::Codegen::LoweringConfigTilingLevelAttr, which is a list of tile sizes
    with optional scalable representation like vector types. E.g.,

      #iree_cpu.lowering_config<
        distribution = [128, 128, 0],
        cache_parallel = [64, 64, 0],
        cache_reduction = [0, 0, 16],
        vector_common_parallel = [[4], [4], 0],
        vector_reduction = [0, 0, [4]],
        vector_inner_parallel = [0, 0, 0]
      >

    For more details, see the implementation in IREECPUAttrs.cpp.

    Note that it is undefined if more than one of vector tiling levels set a
    value on a dimension. They are expected to be disjoint. It is not enforced
    in the verifier, because we want to keep the flexibility when something is
    wrong in a lowering config. E.g., some transformations still work even if
    they are not disjoint.
  }];
  let parameters = (ins
    AttrParameter<"DictionaryAttr",
        "The configured fields, including tiling levels.">:$config
  );
  let builders = [
    AttrBuilder<(ins "llvm::SmallVector<mlir::NamedAttribute>":$items)>,
  ];
  let extraClassDeclaration = [{
    /// Returns the attribute that indicates the tiling sizes.
    static Attribute getTilingLevelAttr(MLIRContext *ctx,
                                   ArrayRef<int64_t> tileSizes);
    static Attribute getTilingLevelAttr(MLIRContext *ctx,
                                   ArrayRef<int64_t> tileSizes,
                                   ArrayRef<bool> scalableFlags);

    /// Returns a vector that contains all the tiling information in the config.
    SmallVector<LoweringConfigLevelInfo> getAvailableTilingInfo();
  }];
  let hasCustomAssemblyFormat = 1;
  let genVerifyDecl = 1;
}

//===----------------------------------------------------------------------===//
// Encoding Resolvers.
//===----------------------------------------------------------------------===//

def IREECPU_CPUEncodingResolverAttr :
    AttrDef<IREECPU_Dialect, "CPUEncodingResolver"> {
  let mnemonic = "cpu_encoding_resolver";
  let summary = [{The encoding layout attribute for CPU backends.}];
  let description = [{
    This attribute can implement any layout interface methods for encoding
    serialization and or materialization, e.g., Encoding::LayoutMaterializerAttr,
    Codegen::PackedLayoutMaterializerAttr, etc. They are implemented through external model
    mechanism See the implementation in compiler/Codegen/ExternalInterfaces/*.
  }];

  let assemblyFormat = "`<` struct(params) `>`";

  let parameters = (ins
    OptionalParameter<"DictionaryAttr", "Executable target configuration. It is "
    "expected to be used in a pass scope, but not the final IR output.">:$configuration
  );
}

def IREECPU_VMVXEncodingResolverAttr :
    AttrDef<IREECPU_Dialect, "VMVXEncodingResolver"> {
  let mnemonic = "vmvx_encoding_resolver";
  let summary = [{The encoding layout attribute for VMVX backend.}];
  let description = [{
    This attribute can implement any layout interface methods for encoding
    serialization and or materialization, e.g., Encoding::LayoutMaterializerAttr,
    Codegen::PackedLayoutMaterializerAttr, etc. They are implemented through external model
    mechanism See the implementation in compiler/Codegen/ExternalInterfaces/*.
  }];

  let assemblyFormat = "`<` struct(params) `>`";

  let parameters = (ins
    OptionalParameter<"DictionaryAttr", "Executable target configuration. It is "
    "expected to be used in a pass scope, but not the final IR output.">:$configuration
  );
}

#endif // IREE_COMPILER_CODEGEN_DIALECT_CPU_IREECPUATTRS

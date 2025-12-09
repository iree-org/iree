// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_CODEGEN_DIALECT_GPU_IREEGPUATTRS
#define IREE_COMPILER_CODEGEN_DIALECT_GPU_IREEGPUATTRS

include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenInterfaces.td"
include "iree/compiler/Codegen/Dialect/GPU/IR/IREEGPUDialect.td"
include "iree/compiler/Codegen/Dialect/GPU/IR/IREEGPUEnums.td"
include "iree/compiler/Codegen/Dialect/GPU/IR/IREEGPUInterfaces.td"
include "mlir/Dialect/Utils/StructuredOpsUtils.td"
include "mlir/Dialect/SCF/IR/DeviceMappingInterface.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/EnumAttr.td"
include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// GPU Specific Lowering Config Attributes
//===----------------------------------------------------------------------===//

def IREEGPU_LoweringConfigAttr :
    AttrDef<IREEGPU_Dialect, "LoweringConfig", [
      DeclareAttrInterfaceMethods<IREECodegen_LoweringConfigAttrInterface, [
        "getWorkgroupTileSizes",
        "getStaticTilingLevelSizes",
        "getTilingLevelSizes",
        "hasTilingLevel",
        "hasWorkgroupTilingLevel",
        "getLoweringStrategy",
        "getWorkgroupReorderingStrategy"
    ]>
    ]> {
  let mnemonic = "lowering_config";
  let summary = [{Drive lowering of an operation for gpu compilation.}];
  let description = [{
    GPU specific implementation of a lowering config. This carries just a
    dictionary attribute to store any relevant fields. This is the simplest
    form of a lowering config, offering flexibility at the cost of structure.
  }];

  let assemblyFormat = "`<` $attributes `>`";

  let parameters = (ins
    AttrParameter<"DictionaryAttr",
        "The configured fields, including tiling levels">:$attributes
  );
}

def IREEGPU_DerivedThreadConfig :
    AttrDef<IREEGPU_Dialect, "DerivedThreadConfig", [
      DeclareAttrInterfaceMethods<IREECodegen_LoweringConfigAttrInterface, [
        "getStaticTilingLevelSizes",
        "getTilingLevelSizes",
        "hasTilingLevel",
      ]>,
      DeclareAttrInterfaceMethods<IREEGPU_PromotionAttr>
    ]> {
  let mnemonic = "derived_thread_config";
  let summary = [{
    Drive lowering of an operation by deriving thread distribution when needed.
  }];
  let description = [{
    Lowering config for a single thread tiling level that is inferred after
    previous (often reduction) levels of tile + fuse. This is intended for
    fused operations where it is much easier to compute the tile sizes to use
    after previous levels of tile + fuse, rather than trying to pre-propagate
    tiling configs.
  }];
  let assemblyFormat = "";
  let parameters = (ins);
}

def IREEGPU_UseGlobalLoadDma :
    AttrDef<IREEGPU_Dialect, "UseGlobalLoadDMA", [
      DeclareAttrInterfaceMethods<IREECodegen_LoweringConfigAttrInterface, [
        "getStaticTilingLevelSizes",
        "getTilingLevelSizes",
        "hasTilingLevel",
      ]>,
      DeclareAttrInterfaceMethods<IREEGPU_PromotionAttr>
    ]> {
  let mnemonic = "use_global_load_dma";
  let summary = [{
    Drive lowering of an operation by using global load DMA.
  }];
  let description = [{
    Lowering config for when using global load DMA is needed. This is intended for
    tagging operations that are known to be able to use global load DMA.

    Tiling sizes are derived from the translation_info's workgroup_size and subgroup_size.
  }];
  let assemblyFormat = "";
  let parameters = (ins);
}

def IREEGPU_PromoteWithCacheSwizzle :
    AttrDef<IREEGPU_Dialect, "PromoteWithCacheSwizzle", [
      DeclareAttrInterfaceMethods<IREEGPU_PromotionAttr, [
        "promoteOperand",
      ]>
    ]> {
  let mnemonic = "promote_with_cache_swizzle";
  let summary = [{
    Indicate promotion of an operand with setting a cache swizzle value.
  }];
  let description = [{
    When promoting, this will create a `linalg.copy` on the input operand,
    and then the primary producing dispatch input has a buffer cast with cache
    swizzle inserted if possible. For example,

    ```
    %0 = tensor_ext.dispatch.tensor.load : tensor<?x4096xf16>
    %1 = linalg.matmul ins(%0, ...)
    ```

    Becomes with `#iree_gpu.promote_with_cache_swizzle<#iree_gpu.derived_thread_config>`

    ```
    %0 = tensor_ext.dispatch.tensor.load : tensor<?x4096xf16>
    %1 = iree_gpu.buffer_resource_cast cache_swizzle(8192)
    %2 = linalg.copy lowering_config = #iree_gpu.derived_thread_config
    %3 = linalg.matmul ins(%2, ...)
    ```
  }];
  let assemblyFormat = "`<` $copy_config `>`";
  let parameters = (ins
    "Attribute":$copy_config
  );
}

//===----------------------------------------------------------------------===//
// GPU Workgroup Processor (WGP) Level Feature/Limit Attributes
//===----------------------------------------------------------------------===//

// This section lists hardware features/limits at a single GPU workgroup
// processor level. Here a GPU workgroup processor means the basic hardware
// functionality unit where a software workgroup is scheduled onto; that is,
// a compute unit for AMD GPUs or a streaming multiprocessor for NVIDIA GPUs.

def IREEGPU_ComputeBitwidthsAttr : EnumAttr<
  IREEGPU_Dialect, IREEGPU_ComputeBitwidths, "compute_bitwidths"> {
  let cppNamespace = "::mlir::iree_compiler::IREE::GPU";
  let assemblyFormat = "$value";
}

def IREEGPU_StorageBitwidthsAttr : EnumAttr<
  IREEGPU_Dialect, IREEGPU_StorageBitwidths, "storage_bitwidths"> {
  let cppNamespace = "::mlir::iree_compiler::IREE::GPU";
  let assemblyFormat = "$value";
}

def IREEGPU_SubgroupOpsAttr : EnumAttr<
  IREEGPU_Dialect, IREEGPU_SubgroupOps, "subgroup_ops"> {
  let cppNamespace = "::mlir::iree_compiler::IREE::GPU";
  let assemblyFormat = "$value";
}

def IREEGPU_DotProductOpsAttr : EnumAttr<
  IREEGPU_Dialect, IREEGPU_DotProductOps, "dotproduct_ops"> {
  let cppNamespace = "::mlir::iree_compiler::IREE::GPU";
  let assemblyFormat = "$value";
}

//===----------------------------------------------------------------------===//
// InnerTiledSemantics
//===----------------------------------------------------------------------===//

def IREEGPU_InnerTiledSemanticsAttr
    : AttrDef<
          IREEGPU_Dialect, "InnerTiledSemantics",
          [DeclareAttrInterfaceMethods<
               IREECodegen_InnerTiledSemanticsAttrInterface, ["getTileShapes",
                                                              "getOpaque"]>,
]> {
  let mnemonic = "mma_semantics";
  let cppNamespace = "::mlir::iree_compiler::IREE::GPU";

  let description = [{
    Attribute describing aspects of MMA semantics that are orthogonal to the
    aspects described in the mma_layout attrs, and that may change during
    lowerings while the mma_layout's typically stay constant. This includes:
    1. Expanded vs opaque tile shapes.
    2. Distributed vs undistributed tile shapes.
  }];

  let parameters = (ins "bool":$distributed, "bool":$opaque);

  let assemblyFormat = "`<` struct(params) `>`";
}

//===----------------------------------------------------------------------===//
// MMA intrinsic
//===----------------------------------------------------------------------===//

class IREEGPU_MmaEnumAttr<EnumAttrInfo enumInfo, string name = "">
  : EnumAttr<IREEGPU_Dialect, enumInfo, name>;

def IREEGPU_MMAIntrinsicAttr
  : IREEGPU_MmaEnumAttr<IREEGPU_MMAIntrinsic, "mma_intrinsic">;

def IREEGPU_MMAAttr : AttrDef<IREEGPU_Dialect, "MMA", [
  DeclareAttrInterfaceMethods<IREEGPU_MmaInterfaceAttr, [
    "getSubgroupSize",
  ]>,
  DeclareAttrInterfaceMethods<IREECodegen_InnerTileDescAttrInterface, [
    "getExpectedNumInputs",
    "getExpectedNumOutputs",
    "verifyIndexingMaps",
    "getUndistributedTileTypes",
    "getDistributedTileTypes",
    "getUndistributedTileDimExpansion",
    "populateOperandOffsetsSizesStrides",
    "getDistributionMappingKind",
    "getDistributionWorkerCount",
    "buildUnderlyingOperations",
  ]>
]> {
  let mnemonic = "mma_layout";
  let cppNamespace = "::mlir::iree_compiler::IREE::GPU";

  let description = [{
    Attribute describing a particular shape of matrix-multiply and accumulate
    instruction. Abstractly, all attributes of this type represent the following
    unit of arithmetic for matrices A, B, and C.

    ```
      C += A x B
    ```

    The |intrinsic| field specifies which particular MMA intrinsic this refers
    to, with each intrinsic implicating a specific MNK shape and operand types.
    See IREEGPUEnums.td for the definition of the intrinsics.

    If set to true, |col_major| indicates that the result should be produced
    column major. This is equivalent to instead computing:

    ```
      C^T += B^T x A^T
    ```
  }];

  let parameters = (ins
    EnumParameter<IREEGPU_MMAIntrinsic>:$intrinsic,
    DefaultValuedParameter<"bool", "false">:$col_major
  );

  let assemblyFormat = "`<` $intrinsic (`,` `col_major` `=` $col_major^)? `>`";

  let builders = [
    AttrBuilder<(ins "MMAIntrinsic":$intrinsic)>
  ];

  let extraClassDeclaration = [{
    int64_t getBlockSize() const;

    SmallVector<VirtualMMAIntrinsic> getVirtualIntrinsics() const;
  }];
}

//===----------------------------------------------------------------------===//
// DataTiled MMA Attributes
//===----------------------------------------------------------------------===//

/// Helper class to define common interface method implementations for the
/// InnerTileDescAttrInterface based on the DataTiledMMAInterfaceAttr.
class IREEGPU_DataTiledMMAInnerTileDescAttr<string mnemonic, list<Trait> traits = []> :
    AttrDef<IREEGPU_Dialect, mnemonic, !listconcat(traits, [
      DeclareAttrInterfaceMethods<IREECodegen_InnerTileDescAttrInterface, [
        "getUndistributedTileTypes",
        "getDistributedTileTypes",
        "getUndistributedTileDimExpansion",
        "populateOperandOffsetsSizesStrides",
        "getDistributionMappingKind",
        "getDistributionWorkerCount",
      ]>,
      DeclareAttrInterfaceMethods<IREEGPU_DataTiledMMAInterfaceAttr, [
        "getTileSwizzle",
        "getTileMNK",
        "getElementTypes",
        "getSubgroupSize",
        "getFlatWorkgroupSize",
      ]>
  ])> {

  let extraClassDefinition = [{
    //========================================================================//
    // Define interface methods for InnerTileDescAttrInterface with shared
    // implementation among DataTiledMMAInterfaceAttr. The implementations
    // are defined in IREEGPUInterfaces.cpp.
    //========================================================================//

    void $cppClass::getUndistributedTileTypes(
        SmallVectorImpl<VectorType>& result) const {
      return cast<DataTiledMMAInterfaceAttr>(*this)
          .getUndistributedTileTypes(result);
    }

    void $cppClass::getDistributedTileTypes(
        SmallVectorImpl<VectorType>& result) const {
      return cast<DataTiledMMAInterfaceAttr>(*this)
          .getDistributedTileTypes(result);
    }

    std::optional<::mlir::SmallVector<int64_t, 2>>
    $cppClass::getUndistributedTileDimExpansion(int64_t operandIndex,
                                                int64_t logicalDim) const {
      return cast<DataTiledMMAInterfaceAttr>(*this)
          .getUndistributedTileDimExpansion(operandIndex, logicalDim);
    }

    LogicalResult $cppClass::populateOperandOffsetsSizesStrides(
        OpBuilder &builder, Location loc, uint32_t operandIndex, Value laneId,
        ArrayRef<int64_t> permutation, SmallVectorImpl<OpFoldResult> &offsets,
        SmallVectorImpl<OpFoldResult> &sizes,
        SmallVectorImpl<OpFoldResult> &strides) const {
      return cast<DataTiledMMAInterfaceAttr>(*this)
          .populateOperandOffsetsSizesStrides(builder, loc, operandIndex,
                                              laneId, permutation, offsets,
                                              sizes, strides);
    }

    Attribute $cppClass::getDistributionMappingKind() const {
      return cast<DataTiledMMAInterfaceAttr>(*this)
          .getDistributionMappingKind();
    }

    OpFoldResult $cppClass::getDistributionWorkerCount(
        OpBuilder &builder, Location loc, Operation *opToDistribute) const {
      return cast<DataTiledMMAInterfaceAttr>(*this)
          .getDistributionWorkerCount(builder, loc, opToDistribute);
    }
  }];
}

def IREEGPU_DataTiledMMAAttr :
    IREEGPU_DataTiledMMAInnerTileDescAttr<"DataTiledMMA", [
  DeclareAttrInterfaceMethods<IREEGPU_MmaInterfaceAttr, [
    "getSubgroupSize",
  ]>,
  DeclareAttrInterfaceMethods<IREECodegen_InnerTileDescAttrInterface, [
    "getExpectedNumInputs",
    "getExpectedNumOutputs",
    "verifyIndexingMaps",
    "buildUnderlyingOperations",
  ]>
]> {
  let mnemonic = "data_tiled_mma_layout";
  let cppNamespace = "::mlir::iree_compiler::IREE::GPU";

  let description = [{
    This mma variant represents MMA ops with data-tiling details. The
    `intrinsic` field specifies which particular MMA intrinsic is targeted by
    the data-tiling.

    The `intrinsics_{m,n,k}` parameters control unrolling along M, N, K
    dimensions. The product of these parameters equals the number of intrinsics
    generated.

    The `operands_interleaving_intrinsics_{m,n,k}` control the tile layout.
    By default, intrinsic-level sub-tiles are contiguous and arranged
    side-by-side in the overall tile layout. When
    `operands_interleaving_intrinsics_{m,n,k}` is non-empty, each entry
    indicates an operand (given by its operand index) for which the
    `intrinsics_{m,n,k}` tiles are interleaved to the innermost positions of
    the entire swizzled layout. For example, when `intrinsics_k = 2` and
    `operands_interleaving_intrinsics_k = [0, 1]`, there are two intrinsics
    along the k-dimension and their layouts are interleaved in operands 0
    and 1, meaning LHS and RHS. When multiple interleavings are applied,
    `intrinsics_k` is placed innermost, then going outward `intrinsics_n`,
    then `intrinsics_m`.

    The `subgroups_{m,n,k}` parameters control the number of subgroups and the
    layout of the tile accesses made by different subgroups. The product of the
    `subgroups_{m,n,k}` equals the number of subgroups, and for instance
    subgroups_m is how many subgroups are working in parallel along the M
    dimension. Code generation supports arbitrary values of subgroups_m and
    subgroups_n. On the other hand, subgroups_k requires custom kernels to
    handle the implied cross-subgroup reduction.
  }];

  let assemblyFormat = "`<` struct(params) `>`";

  let parameters =
      (ins EnumParameter<IREEGPU_MMAIntrinsic>:$intrinsic,
          DefaultValuedParameter<
              "int64_t", "1",
              "Intrinsic count along the M dimension.">:$intrinsics_m,
          DefaultValuedParameter<
              "int64_t", "1",
              "Subgroup count along the M dimension.">:$subgroups_m,
          DefaultValuedParameter<
              "int64_t", "1",
              "Intrinsic count along the N dimension.">:$intrinsics_n,
          DefaultValuedParameter<
              "int64_t", "1",
              "Subgroup count along the N dimension.">:$subgroups_n,
          DefaultValuedParameter<"int64_t", "1",
                                 "Intrinsic count along the K dimension.">:$intrinsics_k,
          DefaultValuedParameter<
              "int64_t", "1",
              "Subgroup count along the K dimension.">:$subgroups_k,
          OptionalParameter<
              "DenseI64ArrayAttr">:$operands_interleaving_intrinsics_m,
          OptionalParameter<
              "DenseI64ArrayAttr">:$operands_interleaving_intrinsics_n,
          OptionalParameter<
              "DenseI64ArrayAttr">:$operands_interleaving_intrinsics_k);
}

def IREEGPU_DataTiledScaledMMAAttr :
    IREEGPU_DataTiledMMAInnerTileDescAttr<"DataTiledScaledMMA", [
  DeclareAttrInterfaceMethods<IREECodegen_InnerTileDescAttrInterface, [
    "getExpectedNumInputs",
    "getExpectedNumOutputs",
    "verifyIndexingMaps",
    "buildUnderlyingOperations"
  ]>
]> {
  let mnemonic = "data_tiled_scaled_mma_layout";
  let cppNamespace = "::mlir::iree_compiler::IREE::GPU";

  let description = [{
    This mma variant represents scaled MMA ops with data-tiling details. The
    |intrinsic| field specifies which particular scaled MMA intrinsic is
    targeted by the data-tiling.

    The |lhs_elem_type|, |rhs_elem_type|, and |acc_elem_type| fields specify the
    element types of the LHS, RHS, and ACC operands respectively. The element
    types for the scales operands are always Float8E8M0FNUType, so they are not
    specified on the attribute.

    The other fields are similar to DataTiledMMAAttr, refer to its documentation.
  }];

  let assemblyFormat = "`<` struct(params) `>`";

  let parameters =
      (ins EnumParameter<IREEGPU_ScaledMMAIntrinsic>:$intrinsic,
          "::mlir::Type":$lhs_elem_type, "::mlir::Type":$rhs_elem_type,
          "::mlir::Type":$acc_elem_type,
          DefaultValuedParameter<
              "int64_t", "1",
              "Intrinsic count along the M dimension.">:$intrinsics_m,
          DefaultValuedParameter<
              "int64_t", "1",
              "Subgroup count along the M dimension.">:$subgroups_m,
          DefaultValuedParameter<
              "int64_t", "1",
              "Intrinsic count along the N dimension.">:$intrinsics_n,
          DefaultValuedParameter<
              "int64_t", "1",
              "Subgroup count along the N dimension.">:$subgroups_n,
          DefaultValuedParameter<"int64_t", "1",
                                 "Intrinsic count along the K dimension.">:$intrinsics_k,
          DefaultValuedParameter<
              "int64_t", "1",
              "Subgroup count along the K dimension.">:$subgroups_k,
          OptionalParameter<
              "DenseI64ArrayAttr">:$operands_interleaving_intrinsics_m,
          OptionalParameter<
              "DenseI64ArrayAttr">:$operands_interleaving_intrinsics_n,
          OptionalParameter<
              "DenseI64ArrayAttr">:$operands_interleaving_intrinsics_k);
}

def IREEGPU_VirtualMMAIntrinsicAttr
  : IREEGPU_MmaEnumAttr<IREEGPU_VirtualMMAIntrinsic, "virtual_mma_intrinsic">;

def IREEGPU_VirtualMMAAttr :
    AttrDef<IREEGPU_Dialect, "VirtualMMA", [
  DeclareAttrInterfaceMethods<IREEGPU_MmaInterfaceAttr, [
    "getSubgroupSize",
  ]>,
  DeclareAttrInterfaceMethods<IREECodegen_InnerTileDescAttrInterface, [
    "getExpectedNumInputs",
    "getExpectedNumOutputs",
    "verifyIndexingMaps",
    "getUndistributedTileTypes",
    "getDistributedTileTypes",
    "getDistributionMappingKind",
    "getDistributionWorkerCount",
    "populateOperandOffsetsSizesStrides",
    "buildUnderlyingOperations",
  ]>
]> {
  let mnemonic = "virtual_mma_layout";
  let cppNamespace = "::mlir::iree_compiler::IREE::GPU";

  let description = [{
    This mma variant represents "virtual" MMA ops that has modification to
    its native layouts by intrinsicsK and/or interleave reads. The |intrinsic|
    field represents different kinds of "Virtual" MMA Ops we found helpful.

    These interleaving and/or unrolling changes in the layout is especially
    useful to coalesce reads from shared memory to register or align layouts
    in a chained-matmul operation.

    If set to true (similar to MMAAttr), |col_major| indicates the computation
    is perfomed as below:

    ```
      C^T += B^T x A^T
    ```
  }];

  let assemblyFormat = "`<` $intrinsic (`,` `col_major` `=` $col_major^)? `>`";

  let parameters = (ins
    EnumParameter<IREEGPU_VirtualMMAIntrinsic>:$intrinsic,
    DefaultValuedParameter<"bool", "false">:$col_major
  );

  let builders = [
    AttrBuilder<(ins "VirtualMMAIntrinsic":$intrinsic)>
  ];
  let extraClassDeclaration = [{
    int64_t getBlockSize() const;

    // Factor to unroll K from native MMA/intrinsic size to virtual size.
    // e.g MFMA_F32_16x16x16 has K of 16, while VMFMA_F32_16x16x32 has K of 32
    // in this example, intrinsicsK = 32/16 = 2.
    int64_t getIntrinsicsK() const;
  }];
}

def IREEGPU_MMAOpsArrayAttr : ArrayOfAttr<
  IREEGPU_Dialect, "MMAOpsArray", "mma_ops", "MMAAttr"> {
  let cppNamespace = "::mlir::iree_compiler::IREE::GPU";
}

def IREEGPU_ScaledMMAOpsArrayAttr : ArrayOfAttr<
  IREEGPU_Dialect, "ScaledMMAOpsArray", "scaled_mma_ops", "ScaledMMAAttr"> {
  let cppNamespace = "::mlir::iree_compiler::IREE::GPU";
}

def IREEGPU_ScaledMMAAttr :
    AttrDef<IREEGPU_Dialect, "ScaledMMA", [
  DeclareAttrInterfaceMethods<IREECodegen_InnerTileDescAttrInterface, [
    "getExpectedNumInputs",
    "getExpectedNumOutputs",
    "verifyIndexingMaps",
    "getUndistributedTileTypes",
    "getDistributedTileTypes",
    "getUndistributedTileDimExpansion",
    "populateOperandOffsetsSizesStrides",
    "getDistributionMappingKind",
    "getDistributionWorkerCount",
    "buildUnderlyingOperations",
  ]>
]> {
  let mnemonic = "scaled_mma_layout";
  let cppNamespace = "::mlir::iree_compiler::IREE::GPU";

  let description = [{
    This attribute represents MMAs that operate on low-precision floating point
    types (like fp4 or fp8) and apply _block scales_ to their inputs,
    where each |blockSize| elements share a scale (which is currently
    always a f32 exponent / f8E8M0).

    The intrinsic is a ScaledMMAIntrinsic, and the element types are given
    on the attribute, not in the intrinsic enum, since these scaled
    operations allow arbitrary combinations of fp4/fp6/fp8 inputs.

    This intrinsic takes its four parameters as (lhs, lhs_scale, rhs, rhs_scale).
  }];

  let assemblyFormat = "`<` struct(params) `>`";

  let parameters = (ins
    EnumParameter<IREEGPU_ScaledMMAIntrinsic>:$intrinsic,
    "::mlir::Type":$lhs_elem_type,
    "::mlir::Type":$rhs_elem_type,
    "::mlir::Type":$acc_elem_type,
    DefaultValuedParameter<"bool", "false", "if this is a column-major scaled MFMA">:$col_major
  );

  let extraClassDeclaration = [{
    // Return all supported element types for inputs
    static SmallVector<Type> getSupportedInputTypes(MLIRContext *ctx);
    static SmallVector<Type> getSupportedOutputTypes(MLIRContext *ctx);

    /// For scaled MFMA intrinsics, the scales are packed into a 32 bit register.
    /// Since scales are 8 bits, the intrinsic takes a vector of 4 scales, along
    /// with a selector to indicate which scale is used.
    int64_t getScalesVectorSize() const {
      return 4;
    }

    // Return the number of elements per shared scale.
    int64_t getBlockSize() const;

    // Return preferred subgroup size
    int64_t getSubgroupSize() const;

    /// Returns the shape of the scaled MMA operation:
    /// ```
    ///   C += A * B
    /// ```
    /// Where, up to some transpositions, `A` has shape `<M, K, KB>`, `B`
    /// has shape `<K, KB, N>`, and `C` has shape `<M, N>`.
    ::std::tuple<int64_t, int64_t, int64_t, int64_t>
    getScaledMNKShape() const {
      ::llvm::SmallVector<::mlir::VectorType> preThreadTypes;
      getUndistributedTileTypes(preThreadTypes);
      ::llvm::ArrayRef<int64_t> accShape = preThreadTypes[4].getShape();
      ::llvm::ArrayRef<int64_t> lhsShape = preThreadTypes[0].getShape();
      return {accShape[0], accShape[1], lhsShape[1], lhsShape[2]};
    }
  }];
}

//===----------------------------------------------------------------------===//
// iree_gpu.gpu_encoding_resolver
//===----------------------------------------------------------------------===//

def IREEGPU_GPUEncodingResolverAttr :
    AttrDef<IREEGPU_Dialect, "GPUEncodingResolver"> {
  let mnemonic = "gpu_encoding_resolver";
  let summary = [{The encoding layout attribute for GPU backend.}];
  let description = [{
    This attribute can implement any layout interface methods for encoding
    serialization and or materialization, e.g., Encoding::LayoutMaterializerAttr,
    Codegen::PackedLayoutMaterializerAttr, etc. They should be implemented through external
    model mechanism because we do not want to relocate domain-specific logic to
    the dialect implementation, and we can have better code structure. See the
    implementation in compiler/Codegen/ExternalInterfaces/*.
  }];

  let assemblyFormat = "`<` struct(params) `>`";

  let parameters = (ins
    OptionalParameter<"DictionaryAttr", "Executable target configuration. It is "
     "expected to be used in a pass scope, but not the final IR output.">:$configuration
  );
}

//===----------------------------------------------------------------------===//
// iree_gpu.gpu_padding_resolver
//===----------------------------------------------------------------------===//

def IREEGPU_GPUPaddingResolverAttr : AttrDef<IREEGPU_Dialect, "GPUPaddingResolver"> {
  let mnemonic = "gpu_padding_resolver";
  let summary = [{The padded encoding layout attribute for GPU targets.}];
  let assemblyFormat = "`<` struct(params) `>`";

  let description = [{
    Describes padding preferences for a given GPU target.
    This attribute can implement any encoding interface for data-tiling,
    e.g., Encoding::LayoutResolverAttr, etc. They should be implemented through
    external model mechanism because we do not want to relocate domain-specific
    logic to the dialect implementation, and we can have better code structure.
    See the implementation in compiler/Codegen/ExternalInterfaces/*.
  }];

  let parameters = (ins
    // Relevant target properties that will later allow us to decide the
    // serialized pad layout.
    OptionalParameter<"std::optional<uint32_t>">:$cache_line_bytes,
    OptionalParameter<"std::optional<uint32_t>">:$cache_sets
  );
}

//===----------------------------------------------------------------------===//
// Workgroup processor level description
//===----------------------------------------------------------------------===//

def IREEGPU_TargetWgpAttr : AttrDef<IREEGPU_Dialect, "TargetWgp"> {
  let summary = [{Workgroup processor level target description.}];
  let description = [{
    This attribute contains hardware features/limits at a single GPU workgroup
    processor (WGP) level. Here a GPU workgroup processor means the basic
    hardware functionality unit where a software workgroup is scheduled onto;
    that is, a compute unit for AMD GPUs or a streaming multiprocessor for
    NVIDIA GPUs.
  }];

  let mnemonic = "target_wgp";
  let cppNamespace = "::mlir::iree_compiler::IREE::GPU";

  let parameters = (ins
    // Features
    "ComputeBitwidthsAttr":$compute,
    "StorageBitwidthsAttr":$storage,
    "SubgroupOpsAttr":$subgroup,
    DefaultValuedParameter<"DotProductOpsAttr", "DotProductOpsAttr::get($_ctxt, DotProductOps::None)">:$dot,
    // MMAs available on this target. Heuristic MMA selection runs is order.
    DefaultValuedParameter<"MMAOpsArrayAttr", "MMAOpsArrayAttr::get($_ctxt, {})">:$mma,
    DefaultValuedParameter<"ScaledMMAOpsArrayAttr", "ScaledMMAOpsArrayAttr::get($_ctxt, {})">:$scaled_mma,

    // Limits
    // Supported subgroup size choices.
    "DenseI32ArrayAttr":$subgroup_size_choices,
    // The maximal number of threads per X/Y/Z dimension in one workgroup.
    "DenseI32ArrayAttr":$max_workgroup_sizes,
    // The maximal number of threads we can have in one workgroup.
    "int32_t":$max_thread_count_per_workgroup,
    // The maximal number of shared memory bytes we can allocate per workgroup.
    "int32_t":$max_workgroup_memory_bytes,
    // The maximum number of workgroups per X/Y/Z dimension in a dispatch.
    "DenseI32ArrayAttr":$max_workgroup_counts,
    // Max load instruction size in bits. TODO(#18849): populate on all GPUs.
    OptionalParameter<"std::optional<int32_t>">:$max_load_instruction_bits,
    // Number of SIMDs per workgroup processor. TODO(#18849): populate on all GPUs.
    OptionalParameter<"std::optional<int32_t>">:$simds_per_wgp,
    // VGPR register space size in bits. TODO(#18849): populate on all GPUs.
    OptionalParameter<"std::optional<int32_t>">:$vgpr_space_bits,
    // Available bit-widths for direct load from global to LDS memory.
    OptionalParameter<"DenseI64ArrayAttr">:$dma_sizes,

    // An optional extra dict
    // This field allows to inject more features/limits not supported in the
    // above list for better flexibility.
    OptionalParameter<"DictionaryAttr">:$extra
  );

  let assemblyFormat = "`<` struct(params) `>`";
}

//===----------------------------------------------------------------------===//
// GPU Chip Level Feature/Limit Attributes
//===----------------------------------------------------------------------===//

// This section lists hardware features/limits at a single GPU chip level.
// Here a GPU chip means the hardware functionality scope where the whole
// software compute grid is scheduled onto. A chip typically contains many
// AMD compute units or NVIDIA streaming multiprocessors; it's the final SKU.

def IREEGPU_TargetChipAttr : AttrDef<IREEGPU_Dialect, "TargetChip"> {
  let summary = [{Chip level target description.}];
  let description = [{
    This attribute contains hardware features/limits at a single GPU chip level.
    Here a GPU chip means the hardware functionality scope where the whole
    software compute grid is scheduled onto. A chip typically contains many
    AMD compute units or NVIDIA streaming multiprocessors; it's the final SKU.
  }];

  let mnemonic = "target_chip";
  let cppNamespace = "::mlir::iree_compiler::IREE::GPU";

  let parameters = (ins
    "uint32_t":$wgp_count,

    // An optional SKU identifier to distinguish different models.
    OptionalParameter<"StringAttr">:$sku,
    // An optional memory bandwidth in TB/s.
    OptionalParameter<"FloatAttr">:$memory_bandwidth_tbps,
    // An optional performance dictionary in TFLOPS.
    OptionalParameter<"DictionaryAttr">:$perf_tflops,
    // An optional extra dict
    // This field allows to inject more features/limits not supported in the
    // above list for better flexibility.
    OptionalParameter<"DictionaryAttr">:$extra
  );

  let assemblyFormat = "`<` struct(params) `>`";
}

//===----------------------------------------------------------------------===//
// GPU Target Attributes
//===----------------------------------------------------------------------===//

def IREEGPU_TargetAttr : AttrDef<IREEGPU_Dialect, "Target", [
  DeclareAttrInterfaceMethods<IREECodegen_TargetInfoAttrInterface, [
    "getMaximumWorkgroupCount"]>]> {
  let summary = [{Full GPU target attribute.}];
  let description = [{
    This attributes describes a full GPU target. It contains a few fields:
    * The canonical target architecture for compilation, e.g., sm_80 for
      cuda, gfx942 for hip
    * A TargetWgpAttr describing the GPU features and limits in a single
      GPU workgroup processor (WGP), that is, AMD compute unit or NVIDIA
      streaming multiprocessor
    * An optional TargetChipAttr describing GPU features for the final chip
      or product, e.g., wgp count
  }];

  let mnemonic = "target";
  let cppNamespace = "::mlir::iree_compiler::IREE::GPU";

  let parameters = (ins
    StringRefParameter<"target architecture">:$arch,
    StringRefParameter<"target features">:$features,
    "TargetWgpAttr":$wgp,
    OptionalParameter<"TargetChipAttr">:$chip
  );

  let assemblyFormat = "`<` struct(params) `>`";

  let extraClassDeclaration = [{
    // Subgroup size related APIs

    int getMinSubgroupSize() const {
      return *llvm::min_element(getWgp().getSubgroupSizeChoices().asArrayRef());
    }
    int getMaxSubgroupSize() const {
      return *llvm::max_element(getWgp().getSubgroupSizeChoices().asArrayRef());
    }
    // Returns the preferred subgroup size. If the target supports multiple
    // subgroup sizes, pick the smallest one.
    //
    // AMD RDNA GPUs supports multiple subgroup sizes and the preferred one
    // differ given the API--HIP prefers 32 while Vulkan prefers 64.
    // We force Vulkan side to use 32 to be consistent with the HIP backend;
    // might have implications on perf.
    int getPreferredSubgroupSize() const {
      return *llvm::min_element(getWgp().getSubgroupSizeChoices().asArrayRef());
    }

    // Hardware feature related APIs

    bool supportsSubgroupShuffle() const {
      return bitEnumContainsAll(getWgp().getSubgroup().getValue(),
                                SubgroupOps::Shuffle);
    }

    // Vendor querying APIs

    bool isAMD() const {
      return getArch().starts_with("gfx") || getArch().starts_with("rdna");
    }
    bool isApple() const { return getArch().starts_with("apple"); }
    bool isARM() const { return getArch().starts_with("valhall"); }
    bool isNVIDIA() const { return getArch().starts_with("sm_"); }
    bool isQualcomm() const { return getArch().starts_with("adreno"); }

    // CUDA specific querying APIs

    std::optional<int> getCUDAComputeCapability() const;
    // Returns true if this target supports TensoreCore MMA ops with TF32
    // input types.
    bool supportsTF32InputMMAOps() const;
    // Returns true if this target supports TensorCore synchronized MMA ops.
    bool supportsSyncMMAOps() const;
  }];
}

//===----------------------------------------------------------------------===//
// GPU Lane ID
//===----------------------------------------------------------------------===//

def IREEGPU_LaneIdAttr : AttrDef<IREEGPU_Dialect, "LaneId", [
      DeclareAttrInterfaceMethods<DeviceMappingAttrInterface>
  ]> {
  let mnemonic = "lane_id";
  let cppNamespace = "::mlir::iree_compiler::IREE::GPU";
  let parameters = (ins
    "int64_t":$dim
  );
  let assemblyFormat = "`<` $dim `>`";
  let description = [{
    An attribute for mapping scf.forall ops to subgroup lanes.
  }];
}

//===---------------------------------------------------------------------===//
// Conditional Transpose Workgroup Reordering
//===---------------------------------------------------------------------===//

def IREEGPU_ConditionalTransposeAttr  :
    AttrDef<IREEGPU_Dialect, "ConditionalTranspose", [
    DeclareAttrInterfaceMethods<IREECodegen_WorkgroupReorderingAttrInterface, [
        "generateLoopHeaderFn",
        "generateLoopTerminatorFn"
      ]>
    ]> {
  let mnemonic = "conditional_transpose";
  let summary = [{Enable or disable transposed workgroup reordering based on the workload.}];
  let description = [{
    This attribute controls conditional transposed workgroup reordering, determined by tile size and
    the number of CUs. This strategy is specifically designed to optimize the ping-pong matmul specialization.
    At each iteration over the K dimension, `X` distinct tiles for RHS and 'Y' distinct tiles for LHS are loaded.
    The strategy compares the sum of X and Y for the default ordering versus the transposed ordering, and selects
    the configuration with the minimum value.
    Calculation breakdown:
     - LHS shape: (M, K)
     - RHS shape: (N, K)
     - Tile dimension: (tileM, tileN)
     - numTileLoadsDefault = min(numCUS, ceildiv(N, tileN)) + ceildiv(numCUS, min(numCUS, ceildiv(N, tileN)))
     - numTileLoadsTransposed = min(numCUS, ceildiv(M, tileM)) + ceildiv(numCUS, min(numCUS, ceildiv(M, tileM)))

     Examples for gfx942 (304 CUs):
     - (M=8192, N=128256, K=4096) (tileM=256, tileN=256)
      - numTileLoadsDefault = 304 + ceildiv(304, 304) = 305
      - numTileLoadsTransposed = 32 + ceildiv(304, 32) = 42
      => numTileLoadsTransposed < numTileLoadsDefault => enable transpose

     - (M=16384, N=8192, K=4096) (tileM=256, tileN=256)
      - numTileLoadsDefault = 32 + ceildiv(304, 32) = 42
      - numTileLoadsTransposed = 64 + ceildiv(304, 64) = 69
      numTileLoadsTransposed > numTileLoadsDefault => disable transpose

      Parameters:
        - numXCDs: Number of XCDs.
        - numCUs: Number of CUs per XCD.
  }];
  let parameters = (ins
    AttrParameter<"int64_t", "">:$numXCDs,
    AttrParameter<"int64_t", "">:$numCUs
  );
  let assemblyFormat = [{
    `<` $numXCDs `,` $numCUs `>`
  }];
  let genVerifyDecl = 1;
}

//===----------------------------------------------------------------------===//
// GPU Pipeline Options
//===----------------------------------------------------------------------===//

def IREEGPU_ReorderWorkgroupsStrategyAttr :
    EnumAttr<IREEGPU_Dialect, IREEGPU_ReorderWorkgroupsStrategy, "reorder_workgroups_strategy"> {
  let assemblyFormat = "`<` $value `>`";
  let cppNamespace = "::mlir::iree_compiler::IREE::GPU";
}

def IREEGPU_GPUPipelineOptionsAttr : AttrDef<IREEGPU_Dialect, "GPUPipelineOptions"> {
  let summary = [{Options attribute for linalg + tensors -> vector + memref GPU pipelines.}];
  let description = [{
    This attributes describes lowering pipeline specific configuration options:
    * prefetch_num_stages: Integer option controlling the software pipelining
      stages for shared memory prefetching. 0 or 1 disables prefetching, 2+
      enables prefetching with that many pipeline stages.
    * no_reduce_shared_memory_bank_conflicts: Boolean option indicating whether
      or not to skip the bank conflict reduction pass in the lowering pipeline.
    * reorder_workgroups_strategy: Enum attribute indicating which strategy to
      choose for the workgroup reordering pass. Options are `None`, `Swizzle`,
      and `Transpose`.
  }];

  let mnemonic = "pipeline_options";
  let cppNamespace = "::mlir::iree_compiler::IREE::GPU";

  let parameters = (ins
    OptionalParameter<"std::optional<int64_t>">:$prefetch_num_stages,
    OptionalParameter<"BoolAttr">:$no_reduce_shared_memory_bank_conflicts,
    OptionalParameter<"BoolAttr">:$use_igemm_convolution,
    OptionalParameter<"ReorderWorkgroupsStrategyAttr">:$reorder_workgroups_strategy
  );

  let builders = [
    AttrBuilder<(ins
        CArg<"unsigned", "0">:$prefetch_num_stages,
        CArg<"bool", "false">:$no_reduce_shared_memory_bank_conflicts,
        CArg<"bool", "false">:$use_igemm_convolution,
        CArg<"std::optional<ReorderWorkgroupsStrategy>", "{}">:$reorder_workgroups_strategy)>
  ];

  let assemblyFormat = "`<` struct(params) `>`";

  let extraClassDeclaration = [{
    // Returns the key name for GPUPipelineOptionsAttr in the translation info
    // config dictionary.
    static StringRef getDictKeyName() {
      return "gpu_pipeline_options";
    }
  }];
}

#endif // IREE_COMPILER_CODEGEN_DIALECT_GPU_IREEGPUATTRS

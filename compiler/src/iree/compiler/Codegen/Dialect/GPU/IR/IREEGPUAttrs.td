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

/// TODO: Iterator type arrays are duplicated across dialects upstream and here.
/// These should be unified somewhere as a common iterator type array attr.
def IREEGPU_IteratorTypeEnum
    : EnumAttr<IREEGPU_Dialect, IteratorType, "iterator_type"> {
    let assemblyFormat = "`<` $value `>`";
}

def IREEGPU_IteratorTypeArrayAttr
    : TypedArrayAttrBase<IREEGPU_IteratorTypeEnum,
                         "Iterator type should be an enum.">;

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
      ]>
    ]> {
  let mnemonic = "lowering_config";
  let summary = "drive lowering of an operation for gpu compilation.";
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
  let extraClassDeclaration = [{
    /// Helper to retrieve a target mma intrinsic if present.
    ::mlir::iree_compiler::IREE::GPU::MmaInterfaceAttr getMmaKind() const;
  }];
}

def IREEGPU_DerivedThreadConfig :
    AttrDef<IREEGPU_Dialect, "DerivedThreadConfig", [
      DeclareAttrInterfaceMethods<IREECodegen_LoweringConfigAttrInterface, [
        "getStaticTilingLevelSizes",
        "getTilingLevelSizes",
        "hasTilingLevel",
      ]>
    ]> {
  let mnemonic = "derived_thread_config";
  let summary = [{
    drive lowering of an operation by deriving thread distribution when needed.
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
// Base MMA vector layout
//===----------------------------------------------------------------------===//

class IREEGPU_MmaVectorLayoutAttr<string attrname, string mmaintrinsic> :
    AttrDef<IREEGPU_Dialect, attrname, [
  DeclareAttrInterfaceMethods<IREEGPU_MmaInterfaceAttr, [
    "getABCElementTypes",
    "getABCVectorTypes",
    "getContractionLayout",
    "getMNKShape",
    "getSubgroupSize",
    "buildMmaOperation",
    "populateOperandOffsetsSizesStrides",
  ]>
]> {
  let cppNamespace = "::mlir::iree_compiler::IREE::GPU";

  string baseDescription = [{
    Attribute describing a particular shape of matrix-multiply and accumulate
    instruction. Abstractly, all attributes of this type represent the following
    unit of arithmetic for matrices A, B, and C.

    ```
      C += A x B
    ```

    Where the shape of matrix `A` is `[m, k]`, `B` is `[k, n]`, and
    `C` is `[m, n]`. This intentionally leaves the layout information abstract
    and uses interface methods to materialize layout information only when
    needed. The shape of the mma intrinsic is stored explicitly here as that
    information is queried frequently.

    The element types for this particular mma intrinsic are |aType|, |bType|,
    and |cType| for matrices `A`, `B`, and `C` respectively.

    ######

  }];


  let parameters = (ins
    mmaintrinsic:$intrinsic,
    "int64_t":$mSize,
    "int64_t":$nSize,
    "int64_t":$kSize,
    "::mlir::Type":$aType,
    "::mlir::Type":$bType,
    "::mlir::Type":$cType
  );
}

//===----------------------------------------------------------------------===//
// MMA intrinsic

class IREEGPU_MmaEnumAttr<EnumAttrInfo enumInfo, string name = "">
  : EnumAttr<IREEGPU_Dialect, enumInfo, name>;

def IREEGPU_MMAIntrinsicAttr
  : IREEGPU_MmaEnumAttr<IREEGPU_MMAIntrinsic, "mma_intrinsic">;

def IREEGPU_MMAAttr : IREEGPU_MmaVectorLayoutAttr<"MMA", "MMAIntrinsicAttr"> {
  let mnemonic = "mma_layout";
  let cppNamespace = "::mlir::iree_compiler::IREE::GPU";

  let description = !strconcat(baseDescription, [{
    This mma variant describes configurations for MMA ops. The |intrinsic|
    field specifies which particular MMA intrinsic this refers to, with each
    intrinsic implicating a specific MNK shape and operand types. The intrinsic
    enum name describes these fields as

    <InputType>_MxNxK_<CType>

    Where the element type for the `A` and `B` matrices are both `InputType`.
  }]);

  let hasCustomAssemblyFormat = 1;

  let skipDefaultBuilders = 1;
  let builders = [
    AttrBuilder<(ins "MMAIntrinsic":$intrinsic)>
  ];

  let extraClassDeclaration = [{
    int64_t getBlockSize() const;
    SmallVector<int64_t> getADataDuplicate() const;
    SmallVector<int64_t> getBDataDuplicate() const;
    SmallVector<int64_t> getCDataDuplicate() const;

    // Partial nested layout for an MMA intrinsic's matrix input/output inside
    // a single subgroup.
    //
    // Note that this is just a container used by the following methods; it can
    // contain both the shape and the order.
    struct SingleSubgroupLayout {
      SmallVector<int64_t, 2> outer;
      SmallVector<int64_t, 2> thread;
      SmallVector<int64_t, 2> element;
    };

    // Returns the A/B/C matrix's partial nested layout shape inside a single
    // subgroup. Shape at each outer/thread/element level is a 2-D value,
    // following canonical matmul order--(M, K) for A, (K, N) for B, and
    // (M, N) for C.
    SingleSubgroupLayout getASingleSubgroupLayoutCount() const;
    SingleSubgroupLayout getBSingleSubgroupLayoutCount() const;
    SingleSubgroupLayout getCSingleSubgroupLayoutCount() const;

    // Returns the A/B/C matrix's partial nested layout order inside a single
    // subgroup. Order at each outer/thread/element level is a 2-value
    // permuation vector, following canonical matmul order--(M, K) for A,
    // (K, N) for B, and (M, N) for C.
    SingleSubgroupLayout getASingleSubgroupLayoutOrder() const;
    SingleSubgroupLayout getBSingleSubgroupLayoutOrder() const;
    SingleSubgroupLayout getCSingleSubgroupLayoutOrder() const;
  }];
}

def IREEGPU_MMAOpsArrayAttr : ArrayOfAttr<
  IREEGPU_Dialect, "MMAOpsArray", "mma_ops", "MMAAttr"> {
  let cppNamespace = "::mlir::iree_compiler::IREE::GPU";
}

//===----------------------------------------------------------------------===//
// MMA schedule

def IREEGPU_MmaScheduleAttr : AttrDef<IREEGPU_Dialect, "MMASchedule"> {
  let mnemonic = "mma_schedule";
  let cppNamespace = "::mlir::iree_compiler::IREE::GPU";

  string description = [{
    A schedule of MMA intrinsic instruction and various levels of tile sizes
    to solve a specific contraction problem.
  }];

  let parameters = (ins
    "::mlir::iree_compiler::IREE::GPU::MmaInterfaceAttr":$intrinsic,
    "int64_t":$subgroup_m_count,
    "int64_t":$subgroup_n_count
  );

  let assemblyFormat = "`<` struct(params) `>`";

  let extraClassDeclaration = [{
    // Returns the A/B/C matrix concrete layout targeting |contractOp|.
    ::mlir::FailureOr<::std::tuple<VectorExt::VectorLayoutInterface,
                                 VectorExt::VectorLayoutInterface,
                                 VectorExt::VectorLayoutInterface>>
      getContractionLayout(::mlir::vector::ContractionOp contractOp) const;
  }];
}

//===----------------------------------------------------------------------===//
// Workgroup processor level description

def IREEGPU_TargetWgpAttr : AttrDef<IREEGPU_Dialect, "TargetWgp"> {
  let summary = "Workgroup processor level target description";
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
    "DotProductOpsAttr":$dot,
    "MMAOpsArrayAttr":$mma,

    // Limits
    // Supported subgroup size choices.
    "DenseI32ArrayAttr":$subgroup_size_choices,
    // The maximal number of threads per X/Y/Z dimension in one workgroup.
    "DenseI32ArrayAttr":$max_workgroup_sizes,
    // The maximal number of threads we can have in one workgroup.
    "uint32_t":$max_thread_count_per_workgroup,
    // The maximal number of shared memory bytes we can allocate per workgroup.
    "uint32_t":$max_workgroup_memory_bytes,

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
  let summary = "Chip level target description";
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

def IREEGPU_TargetAttr : AttrDef<IREEGPU_Dialect, "Target"> {
  let summary = "Full GPU target attribute";
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
    int getPreferredSubgroupSize() const {
      return getWgp().getSubgroupSizeChoices().asArrayRef().front();
    }

    bool supportsSubgroupShuffle() const {
      return bitEnumContainsAll(getWgp().getSubgroup().getValue(),
                                SubgroupOps::Shuffle);
    }

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

#endif // IREE_COMPILER_CODEGEN_DIALECT_GPU_IREEGPUATTRS

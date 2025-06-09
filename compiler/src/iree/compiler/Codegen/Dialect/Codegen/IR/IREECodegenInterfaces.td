// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_CODEGEN_DIALECT_CODEGEN_IREECODEGENINTERFACES
#define IREE_COMPILER_CODEGEN_DIALECT_CODEGEN_IREECODEGENINTERFACES

include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenDialect.td"

def IREECodegen_InnerTileDescAttrInterface :
  AttrInterface<"InnerTileDescAttrInterface"> {
  let cppNamespace = "::mlir::iree_compiler::IREE::Codegen";
  let description = [{
    Attribute interface used to descripe an inner tiled operation, such as
    a (sequence of) matrix multiplications that are performed using
    intrinsics which have their own inherent tiling.

    Note: this interface is a generalization of MmaInterfaceAttr.
  }];

  let methods = [
    InterfaceMethod<
      /*desc=*/[{
        Returns the number of non-accumulator input operands  this inner tiled
        operation requires.
      }],
      /*retTy=*/"int64_t",
      /*methodName=*/"getExpectedNumInputs",
      /*args=*/(ins)
    >,
    InterfaceMethod<
      /*desc=*/[{
        If this inner tiled operation imposes additional requirements on
        its indexing maps other than the general ones (todo describe),
        verify them.
      }],
      /*retType=*/"::mlir::LogicalResult",
      /*methodName=*/"verifyIndexingMaps",
      /*args=*/(ins "::llvm::ArrayRef<::mlir::AffineMap>":$maps),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        return ::mlir::success();
      }]
    >,
    InterfaceMethod<
      /*desc=*/[{
        For each operand (including the accumulator, which is the final
        operand), return the vector type of the inner tile before the operation
        is distributed (if applicable). This should be the logical type of
        the operands, ignoring any permutations that may be present on the
        inner_tiled operation.

        For example, a matrix multiplication that takes a MxK inner tile of A,
        a KxN inner tile of B, and accumulates into a MxN inner tile of C
        should return:
        ```
          vector<MxKxtA>, vector<KxNxtB>, vector<MxNxtC>
        ```
        where `tA`, `tB`, and `tC` are the element types of matrices A, B, and C,
        respectively.

        The returned values are placed into the `result` vector, which will have
        its existing contents overwritten if there is any.
      }],
      /*retType=*/"void",
      /*methodName=*/"getUndistributedTileTypes",
      /*args=*/(ins "::llvm::SmallVectorImpl<::mlir::VectorType>&":$result)
    >,
    InterfaceMethod<
      /*desc=*/[{
        For each operand (including the accumulator, which is the final
        operand), return the vector type of the inner tile after the operation
        is distributed to threads. Vectors of this type will be stored into or
        loaded from registers by inndividual threads.

        The returned values are placed into the `result` vector, which will have
        its existing contents overwritten if there is any.
      }],
      /*retType=*/"void",
      /*methodName=*/"getDistributedTileTypes",
      /*args=*/(ins "::llvm::SmallVectorImpl<::mlir::VectorType>&":$result)
    >,
    InterfaceMethod<
      /*desc=*/[{
        Constructs the offsets/sizes/strides for extracting the per-thread
        slice of the `operandIndex`th operand of this inner tiled operation.

        This method is only applicable when distribution is being performed.
      }],
      /*retTy=*/"::mlir::LogicalResult",
      /*methodName=*/"populateOperandOffsetsSizesStrides",
      /*args=*/(ins
        "::mlir::OpBuilder&":$builder,
        "::mlir::Location":$loc,
        "uint32_t":$operandIndex,
        "::mlir::Value":$lane_id,
        "::llvm::ArrayRef<int64_t>":$permutation,
        "::llvm::SmallVectorImpl<::mlir::OpFoldResult>&":$offsets,
        "::llvm::SmallVectorImpl<::mlir::OpFoldResult>&":$sizes,
        "::llvm::SmallVectorImpl<::mlir::OpFoldResult>&":$strides
      ),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        return ::mlir::failure();
      }]
    >,
    InterfaceMethod<
      /*desc=*/[{
        If this intrinsic can be mapped from its undistributed to its distributed
        form using an `scf.forall`, return the attribute describing the mapping
        kind that is to be used.

        If such a mapping is not supported, return Attribute() (the null attribute).
      }],
      /*retTy=*/"::mlir::Attribute",
      /*methodName=*/"getDistributionMappingKind",
      /*args=*/(ins),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        return ::mlir::Attribute();
      }]
    >
  ];

  let extraSharedClassDeclaration = [{
    /// Return the expected size of the undistributed inner tile for each
    /// operand into `results`.
    void getUndistributedOperandSize(::llvm::SmallVectorImpl<int64_t>& result) const {
      result.clear();
      ::llvm::SmallVector<::mlir::VectorType> undistributedTypes;
      $_attr.getUndistributedTileTypes(undistributedTypes);
      ::llvm::append_range(result, ::llvm::map_range(undistributedTypes,
        [](auto vt) { return vt.getNumElements(); }));
    }

    /// Return the element element type of each operand into `results`.
    void getElementTypes(::llvm::SmallVectorImpl<::mlir::Type>& result) const {
      result.clear();
      ::llvm::SmallVector<::mlir::VectorType> undistributedTypes;
      $_attr.getUndistributedTileTypes(undistributedTypes);
      ::llvm::append_range(result, ::llvm::map_range(undistributedTypes,
        [](auto vt) { return vt.getElementType(); }));
    }
  }];
}

def IREECodegen_LoweringConfigAttrInterface :
  AttrInterface<"LoweringConfigAttrInterface"> {
  let cppNamespace = "::mlir::iree_compiler::IREE::Codegen";
  let description = [{
    Attribute interface for specifying information used by the compiler to
    translate a specific operation. The way this information is used is backend
    and/or pipeline specific, so this interface only exposes information needed
    for shared use cases.
  }];

  let methods = [
    InterfaceMethod<
      /*desc=*/[{
        Returns the tile sizes to use for workgroup distribution.

        TODO: This should be queried/constructed from device information and
        used to compute workgroup size.
      }],
      /*retTy=*/"::llvm::SmallVector<int64_t>",
      /*methodName=*/"getWorkgroupTileSizes",
      /*args=*/(ins),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        return ::llvm::SmallVector<int64_t>();
      }]
    >,
    InterfaceMethod<
      /*desc=*/[{
        Returns the loop interchange to use for workgroup distribution.
      }],
      /*retTy=*/"::llvm::SmallVector<int64_t>",
      /*methodName=*/"getWorkgroupInterchange",
      /*args=*/(ins),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        return ::llvm::SmallVector<int64_t>();
      }]
    >,
    InterfaceMethod<
      /*desc=*/[{
        Returns true if the lowering config specifies tile sizes for the given
        tiling level.
      }],
      /*retTy=*/"bool",
      /*methodName=*/"hasTilingLevel",
      /*args=*/(ins "unsigned":$level),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        return false;
      }]
    >,
    InterfaceMethod<
      /*desc=*/[{
        Returns true if the lowering config specifies tile sizes for the
        workgroup tiling level.
      }],
      /*retTy=*/"bool",
      /*methodName=*/"hasWorkgroupTilingLevel",
      /*args=*/(ins),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        return false;
      }]
    >,
    InterfaceMethod<
      /*desc=*/[{
        Returns the tile sizes for the specified tiling level. The
        interpretation of |level| is attribute and backend dependent. The
        |target| is the operation this lowering configuration annotates.

        returns an empty list if sizes are not specified for this level. dynamic
        sizes are specified with `shapedtype::kdynamicsize`.
      }],
      /*retTy=*/"::llvm::SmallVector<int64_t>",
      /*methodName=*/"getStaticTilingLevelSizes",
      /*args=*/(ins
        "unsigned":$level,
        "::mlir::Operation *":$target
      ),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        return ::llvm::SmallVector<int64_t>();
      }]
    >,
    InterfaceMethod<
      /*desc=*/[{
        Constructs the tile sizes for the specified level. The
        interpretation of |level| is attribute and backend dependent. The
        |target| is the operation this lowering configuration annotates.

        Returns an empty list if sizes are not specified for this level.
      }],
      /*retTy=*/"::llvm::SmallVector<::mlir::OpFoldResult>",
      /*methodName=*/"getTilingLevelSizes",
      /*args=*/(ins
        "::mlir::OpBuilder &":$builder,
        "unsigned":$level,
        "::mlir::Operation *":$target
      ),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        return ::llvm::SmallVector<OpFoldResult>();
      }]
    >,
    InterfaceMethod<
      /*desc=*/[{
        Gets the name of the custom lowering strategy to apply to the annotated
        operation.
      }],
      /*retTy=*/"::std::optional<::llvm::StringRef>",
      /*methodName=*/"getLoweringStrategy",
      /*args=*/(ins),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        return std::nullopt;
      }]
    >
  ];
}

def IREECodegen_PackedLayoutMaterializerAttr :
  AttrInterface<"PackedLayoutMaterializerAttr"> {
  let cppNamespace = "::mlir::iree_compiler::IREE::Codegen";
  let description = [{
    An interface that collects a set of methods for packed encoding materialization.
    This helps generalizing materialization for encodings that perform some kind of
    packing and/or swizzling.
  }];

  let methods = [
    InterfaceMethod<
      /*desc=*/[{
        Returns the layout of materialized encoding for a tensor type.
      }],
      /*retTy=*/"::mlir::iree_compiler::IREE::Codegen::MaterializeEncodingInfo",
      /*methodName=*/"getEncodingInfo",
      /*args=*/(ins "::mlir::RankedTensorType":$type),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        assert(false && "unimplemented interface method");
        return MaterializeEncodingInfo{};
      }]
    >
  ];
}

def IREECodegen_SwizzleAttrInterface :
  AttrInterface<"SwizzleAttrInterface"> {
  let cppNamespace = "::mlir::iree_compiler::IREE::Codegen";
  let description = [{
    An interface that describes 1D memref swizzling patterns.
  }];

  let methods = [
    InterfaceMethod<
      /*desc=*/[{
        Swizzles |offset| into memref |src|.
      }],
      /*retTy=*/"::mlir::OpFoldResult",
      /*methodName=*/"swizzleOffset",
      /*args=*/(ins "::mlir::OpBuilder&":$b,
                    "::mlir::Location":$loc,
                    "::mlir::OpFoldResult":$offset,
                    "::mlir::Value":$src),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        return offset;
      }]
    >,
    InterfaceMethod<
      /*desc=*/[{
        Returns the number of elements that remain contiguous with the swizzling
        pattern. This unrolls all accesses to this element count.

        Currently swizzling is only supported if all accesses are multiples of
        this value.

        TODO: Support non-width aligned swizzling.
      }],
      /*retTy=*/"int64_t",
      /*methodName=*/"getAccessElementCount",
      /*args=*/(ins)
    >
  ];
}

def IREECodegen_AnySwizzleAttr : Attr<Or<[
  CPred<"isa<IREE::Codegen::SwizzleAttrInterface>($_self)">,
]>, "swizzling descriptor attributes"> {
  let storageType = [{ IREE::Codegen::SwizzleAttrInterface }];
  let returnType = [{ IREE::Codegen::SwizzleAttrInterface }];
  let convertFromStorage = "$_self";
}

#endif // IREE_COMPILER_CODEGEN_DIALECT_CODEGEN_IREECODEGENINTERFACES

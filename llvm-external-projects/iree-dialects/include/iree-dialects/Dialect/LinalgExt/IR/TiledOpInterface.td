// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_DIALECT_LINALGEXT_TILEDOPINTERFACE
#define IREE_DIALECT_LINALGEXT_TILEDOPINTERFACE

include "mlir/IR/OpBase.td"

def TiledOpInterface : OpInterface<"TiledOpInterface"> {
  let description = [{
    Interface for allowing operations to expose information needed to
    tile it (similar to LinalgOp, but without having access to
    indexing maps)
  }];
  let cppNamespace = "::mlir::iree_compiler::IREE::LinalgExt";
  let methods = [
      InterfaceMethod<
        /*desc=*/[{
          Returns the destination operands. For op with `memref`
          operands, this is the result buffers. For op with `tensor`
          operands, this is the operands that contain the initial
          value for the result. These are "tied" to the result
          buffers. For example, for a `LinalgOp`/`LinalgExt` ops, it
          is the `outs` parameters. For `tensor.insert_slice`, it is
          the `dest` parameter.
        }],
        /*retType=*/"SmallVector<Value>",
        /*methodName=*/"getDestinationOperands",
        /*args=*/(ins "OpBuilder &":$b),
        /*methodBody=*/"",
        /*defaultImplementation=*/"return ValueRange{};"
      >,
      InterfaceMethod<
        /*desc=*/[{
          Returns a list of `StringRef`s that describe the number of
          loops and the iterator types of the operation. The list is
          expected to use
          `getParallelIteratorTypeName()`/`getReductionIteratorTypeName()`
          from MLIR Structured Op Utils.
        }],
        /*retType=*/"SmallVector<StringRef>",
        /*methodName=*/"getLoopIteratorTypes"
      >,
      InterfaceMethod<
        /*desc=*/[{
          Returns a list of ranges that describe the loop bounds and
          step for the loops of the operation.
        }],
        /*retTy=*/"SmallVector<Range>",
        /*methodName=*/"getIterationDomain",
        /*args=*/(ins "OpBuilder &":$b)
      >,
      InterfaceMethod<
        /*desc=*/[{
          Return the loops that are to be distributed given the
          maximum amount of logical processor dimensions available.
        }],
        /*retTy=*/"SmallVector<unsigned>",
        /*methodName=*/"getPartitionableLoops",
        /*args=*/(ins "unsigned ":$maxNumParallelDims),
        /*methodBody=*/"",
        /*defaultImplementation=*/[{
           SmallVector<unsigned> partitionableLoops;
           auto interfaceOp = cast<TiledOpInterface>($_op.getOperation());
           for (auto iteratorType :
               llvm::enumerate(interfaceOp.getLoopIteratorTypes())) {
             if (iteratorType.value() != getParallelIteratorTypeName()) {
               break;
             }
             partitionableLoops.push_back(iteratorType.index());
           }
           if (partitionableLoops.size() > maxNumParallelDims) {
             partitionableLoops.erase(
               partitionableLoops.begin(),
               std::next(partitionableLoops.begin(),
                   partitionableLoops.size() - maxNumParallelDims));
           }
           return partitionableLoops;
        }]
      >,
      InterfaceMethod<
        /*desc=*/[{
          Generates a tiled version of the operation given the tile
          size for the loops.

          Returns the tiled operation generated. If the operation has
          tensor semantics then the result of the tiled values are to
          be inserted into the `outputs` and return in `results`.
        }],
        /*retType=*/"Operation *",
        /*methodName=*/"getTiledImplementation",
        /*args=*/(ins
            "OpBuilder &":$b,
            "ValueRange ":$outputs,
            "ArrayRef<OpFoldResult> ":$offsets,
            "ArrayRef<OpFoldResult> ":$sizes,
            "SmallVectorImpl<Value> &":$results),
        /*methodBody=*/"",
        /*defaultImplementation=*/[{
          return nullptr;
        }]
      >,
      InterfaceMethod<
        /*desc=*/[{
          Generates the loop body implementation. Assume that all the parallel
          loops and reduction loops are created and the insertion point of the
          build is set to the innermost of the loop. This method implements the
          loop body IRs.
        }],
        /*retType=*/"LogicalResult",
        /*methodName=*/"generateScalarImplementation",
        /*args=*/(ins
            "OpBuilder &":$b,
            "Location ":$loc,
            "ValueRange ":$ivs),
        /*methodBody=*/"",
        /*defaultImplementation=*/[{
          return failure();
        }]
      >,
      InterfaceMethod<
        /*desc=*/[{
          Method to return the position of the result tile computed by the tiled operation.

          Specifies what tile of the result of the original tensor is computed
          by the tiled implementation. Expects the same `offsets` and `sizes` as
          used to obtain the tiled implementation of the operation.
        }],
        /*retType=*/"LogicalResult",
        /*methodName=*/"getResultTilePosition",
        /*args=*/(ins
          "OpBuilder &":$b,
          "unsigned":$resultNumber,
          "ArrayRef<OpFoldResult> ":$offsets,
          "ArrayRef<OpFoldResult> ":$sizes,
          "SmallVector<OpFoldResult> &":$resultOffsets,
          "SmallVector<OpFoldResult> &":$resultSizes),
        /*methodBody=*/"",
        /*defaultImplementation=*/[{
          return failure();
        }]
      >,
      InterfaceMethod<
        /*desc=*/[{
          Method to generate the code that produces a tile of the result.

          Generates the IR that computes the tile of a result of the
          operation.  The `offsets` and `sizes` describe the tile of
          the output required. This is different from
          `getTiledImplementation` which generates the tiled
          implementation of the operation given a tile of the
          iteration space. This method generates a tiled
          implementation of the operation based on the tile of the
          result required. This method enables fusion by using tile
          and fuse. The method returns failure if the operation can't be
          tiled to generate the result tile. In practical terms this
          implies it cannot be tiled and fused with its consumers.        

          - `dest` are the Value into which the result of the tiled
            operation is to be inserted into. The type of the `dest`
            Values is same as the types returned by
            `getDestinationOperands` method.
          - `offsets` provides the offset of the tile within the
            iteration space
          - `sizes` provides the size of the tile.
          - `tileDestOperands` specifies whether to also tile `dest` operands
            or not. Avoiding tiling `dest` operands can be useful for 
            composition with various looping container ops.
        }],
        /*retType=*/"FailureOr<Value>",
        /*methodName=*/"generateResultTileValue",
        /*args=*/(ins
          "OpBuilder &":$b,
          "unsigned":$resultNumber,
          "ValueRange":$dest,
          "ArrayRef<OpFoldResult>":$offsets,
          "ArrayRef<OpFoldResult>":$sizes,
          "bool":$tileDestOperands),
        /*methodBody=*/"",
        /*defaultImplementation=*/[{
          return failure();
        }]
      >
  ];
}

#endif  // IREE_DIALECT_LINALGEXT_TILEDOPINTERFACES

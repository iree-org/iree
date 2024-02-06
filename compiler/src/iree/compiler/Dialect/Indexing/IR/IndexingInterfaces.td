// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_DIALECT_INDEXING_IR_INDEXING_INTERFACES
#define IREE_DIALECT_INDEXING_IR_INDEXING_INTERFACES

include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// IREE::Indexing::StaticBoundsOpInterface
//===----------------------------------------------------------------------===//

def Indexing_StaticBoundsOpInterface : OpInterface<"StaticBoundsOpInterface"> {
  let cppNamespace = "::mlir::iree_compiler::IREE::Indexing";

  let description = [{
    Interface for ops to implement propagation of static integer range
    statistics.
  }];

  let methods = [
    InterfaceMethod<
      /*desc=*/[{
        Takes a list of static ranges for each operand or dynamic operand dim
        and returns new static bounds for the given scalar integer or index
        value. The value can either be an op result, or a block argument for
        RegionBranchOpInterfaces.
      }],
      /*retTy=*/"SaturatedIndexRange",
      /*methodName=*/"getIndexRange",
      /*args=*/(ins "Value":$target,
                    "ArrayRef<SaturatedValueRange>":$operandRanges),
      /*methodBody=*/[{}],
      /*defaultImplementation=*/[{
        return SaturatedIndexRange();
      }]
    >,
    InterfaceMethod<
      /*desc=*/[{
        Takes a list of static ranges for each operand or dynamic operand dim
        and returns new static bounds for the dynamic dims of the given shaped
        value. The value can either be an op result, or a block argument for
        RegionBranchOpInterfaces.
      }],
      /*retTy=*/"SaturatedIndexRangeList",
      /*methodName=*/"getDynamicDimRanges",
      /*args=*/(ins "Value":$target,
                    "ArrayRef<SaturatedValueRange>":$operandRanges),
      /*methodBody=*/[{}],
      /*defaultImplementation=*/[{
        ShapedType shapedType = llvm::cast<ShapedType>(target.getType());
        SmallVector<SaturatedIndexRange> ranges;
        for (auto size : shapedType.getShape()) {
          if (ShapedType::isDynamic(size)) {
            ranges.push_back(SaturatedIndexRange());
          }
        }
        return ranges;
      }]
    >,
    InterfaceMethod<
      /*desc=*/[{
        Returns a constant range for a scalar integer or index value.
      }],
      /*retTy=*/"std::optional<SaturatedIndexRange>",
      /*methodName=*/"initializeRange",
      /*args=*/(ins "Value":$target, "bool &":$isFixedPoint),
      /*methodBody=*/[{}],
      /*defaultImplementation=*/[{
        isFixedPoint = false;
        return std::nullopt;
      }]
    >,
    InterfaceMethod<
      /*desc=*/[{
        Returns a constant range for the dynamic dimensions of a shaped value.
      }],
      /*retTy=*/"std::optional<SaturatedIndexRangeList>",
      /*methodName=*/"initializeDimRanges",
      /*args=*/(ins "Value":$target),
      /*methodBody=*/[{}],
      /*defaultImplementation=*/[{
        return std::nullopt;
      }]
    >,
  ];
}

#endif  // IREE_DIALECT_INDEXING_IR_INDEXING_INTERFACES

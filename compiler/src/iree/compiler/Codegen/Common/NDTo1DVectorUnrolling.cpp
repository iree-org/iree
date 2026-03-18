// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===- NDTo1DVectorUnrolling.cpp ------------------------------------------===//
//
// Unrolls ND static vector types into multiple 1D vectors at SCF region
// boundaries using 1:N type conversion. Each ND vector is decomposed into
// slices along the innermost dimension.
//
// Example: vector<4x8xf32> -> vector<8xf32> x4
//
//   Before:
//     %0 = scf.for ... iter_args(%arg = %init) -> vector<4x8xf32> {
//       scf.yield %arg : vector<4x8xf32>
//     }
//
//   After:
//     %e0 = vector.extract %init[0] : vector<8xf32> from vector<4x8xf32>
//     %e1 = vector.extract %init[1] : vector<8xf32> from vector<4x8xf32>
//     %e2 = vector.extract %init[2] : vector<8xf32> from vector<4x8xf32>
//     %e3 = vector.extract %init[3] : vector<8xf32> from vector<4x8xf32>
//     %0:4 = scf.for ... iter_args(%a0 = %e0, %a1 = %e1, %a2 = %e2,
//                                   %a3 = %e3)
//         -> (vector<8xf32>, vector<8xf32>, vector<8xf32>, vector<8xf32>) {
//       scf.yield %a0, %a1, %a2, %a3 : vector<8xf32>, ...
//     }
//     %p = ub.poison : vector<4x8xf32>
//     %i0 = vector.insert %0#0, %p [0] : vector<8xf32> into vector<4x8xf32>
//     ...
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/SCF/Transforms/Patterns.h"
#include "mlir/Dialect/UB/IR/UBOps.h"
#include "mlir/Dialect/Utils/IndexingUtils.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir::iree_compiler {

#define GEN_PASS_DEF_NDTO1DVECTORUNROLLINGPASS
#include "iree/compiler/Codegen/Common/Passes.h.inc"

namespace {

struct UnrollVectorTypeConverter final : public TypeConverter {
  UnrollVectorTypeConverter() {
    addConversion([](Type type) -> std::optional<Type> { return type; });

    addConversion([](VectorType type, SmallVectorImpl<Type> &types)
                      -> std::optional<LogicalResult> {
      if (type.getRank() <= 1) {
        types.push_back(type);
        return success();
      }
      assert(type.hasStaticShape() &&
             "expected static shape for ND to 1D vector unrolling");
      int64_t innerDim = type.getShape().back();
      int64_t num1DVectors = type.getNumElements() / innerDim;
      Type innerDimVector = VectorType::get({innerDim}, type.getElementType());
      types.append(num1DVectors, innerDimVector);
      return success();
    });

    addSourceMaterialization([](OpBuilder &builder, VectorType targetType,
                                ValueRange inputs, Location loc) -> Value {
      Value result = ub::PoisonOp::create(builder, loc, targetType);
      SmallVector<int64_t> iteratorSpace(targetType.getShape().drop_back());
      for (auto [input, idx] : llvm::zip_equal(
               inputs, StaticTileOffsetRange(
                           iteratorSpace,
                           SmallVector<int64_t>(iteratorSpace.size(), 1)))) {
        result = vector::InsertOp::create(builder, loc, input, result, idx);
      }
      return result;
    });

    addTargetMaterialization([](OpBuilder &builder, TypeRange targetTypes,
                                ValueRange sources, Location loc,
                                Type originalType) -> SmallVector<Value> {
      assert(sources.size() == 1 && "expected single source value");
      Value source = sources[0];
      SmallVector<Value> results;
      VectorType originalTypeVec = cast<VectorType>(originalType);
      SmallVector<int64_t> iteratorSpace(
          originalTypeVec.getShape().drop_back());
      for (SmallVector<int64_t> idx : StaticTileOffsetRange(
               iteratorSpace, SmallVector<int64_t>(iteratorSpace.size(), 1))) {
        results.push_back(vector::ExtractOp::create(builder, loc, source, idx));
      }
      return results;
    });
  }
};

struct NDTo1DVectorUnrollingPass final
    : impl::NDTo1DVectorUnrollingPassBase<NDTo1DVectorUnrollingPass> {

  void runOnOperation() override {
    MLIRContext *ctx = &getContext();
    RewritePatternSet patterns(ctx);

    UnrollVectorTypeConverter typeConverter;
    ConversionTarget target(*ctx);

    scf::populateSCFStructuralTypeConversionTarget(typeConverter, target);
    scf::populateSCFStructuralTypeConversions(typeConverter, patterns);

    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

} // namespace

} // namespace mlir::iree_compiler

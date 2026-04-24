// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <gtest/gtest.h>

#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenAttrs.h"
#include "iree/compiler/Codegen/Dialect/GPU/IR/IREEGPUAttrs.h"
#include "iree/compiler/Codegen/Dialect/GPU/IR/IREEGPUDialect.h"
#include "iree/compiler/Codegen/Dialect/GPU/TargetUtils/KnownTargets.h"
#include "iree/compiler/Codegen/LLVMGPU/LLVMGPUConstraintGenerator.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/MLIRContext.h"

namespace mlir::iree_compiler {
namespace {

using IntKnobAttr = IREE::Codegen::IntKnobAttr;
using OneOfKnobAttr = IREE::Codegen::OneOfKnobAttr;

// Shared fixture providing MLIRContext, ModuleOp, and helpers to
// construct common linalg test ops.
class LinalgTestBase : public ::testing::Test {
protected:
  LinalgTestBase() {
    DialectRegistry reg;
    reg.insert<arith::ArithDialect, func::FuncDialect, linalg::LinalgDialect,
               tensor::TensorDialect, IREE::GPU::IREEGPUDialect>();
    ctx.appendDialectRegistry(reg);
    ctx.loadAllAvailableDialects();
  }

  MLIRContext *getContext() { return &ctx; }

  OpBuilder setupBuilder() {
    OpBuilder builder(&ctx);
    module = ModuleOp::create(builder.getUnknownLoc());
    builder.setInsertionPointToStart(module->getBody());
    return builder;
  }

  linalg::MatmulOp getTestMatmulOp(Type inputType, Type accType,
                                   unsigned int m = 16, unsigned int n = 16,
                                   unsigned int k = 32) {
    OpBuilder builder = setupBuilder();
    Location loc = builder.getUnknownLoc();

    auto lhsType = RankedTensorType::get({m, k}, inputType);
    auto rhsType = RankedTensorType::get({k, n}, inputType);
    auto accTy = RankedTensorType::get({m, n}, accType);

    Value lhs =
        tensor::EmptyOp::create(builder, loc, lhsType.getShape(), inputType);
    Value rhs =
        tensor::EmptyOp::create(builder, loc, rhsType.getShape(), inputType);
    Value acc =
        tensor::EmptyOp::create(builder, loc, accTy.getShape(), accType);

    return linalg::MatmulOp::create(builder, loc, TypeRange{accTy},
                                    ValueRange{lhs, rhs}, ValueRange{acc});
  }

  linalg::MatmulOp getTestMatmulOp(unsigned int m = 16, unsigned int n = 16,
                                   unsigned int k = 32) {
    return getTestMatmulOp(Float16Type::get(&ctx), Float32Type::get(&ctx), m, n,
                           k);
  }

  linalg::FillOp getTestFillOp() {
    OpBuilder builder = setupBuilder();
    Location loc = builder.getUnknownLoc();

    auto f32 = builder.getF32Type();
    auto outType = RankedTensorType::get({16, 16}, f32);

    Value val = arith::ConstantOp::create(builder, loc, f32,
                                          builder.getF32FloatAttr(0.0));

    Value out = tensor::EmptyOp::create(builder, loc, outType.getShape(),
                                        outType.getElementType());

    return linalg::FillOp::create(builder, loc, TypeRange{outType},
                                  ValueRange{val}, ValueRange{out});
  }

  linalg::Conv2DNhwcHwcfOp
  getTestConv2DNhwcHwcfOp(unsigned int b = 1, unsigned int ih = 3,
                          unsigned int iw = 3, unsigned int ic = 1,
                          unsigned int kh = 1, unsigned int kw = 1,
                          unsigned int oc = 1) {
    OpBuilder builder = setupBuilder();
    Location loc = builder.getUnknownLoc();
    auto f32 = builder.getF32Type();

    unsigned int oh = ih - kh + 1;
    unsigned int ow = iw - kw + 1;

    auto inputTy = RankedTensorType::get({b, ih, iw, ic}, f32);
    auto filterTy = RankedTensorType::get({kh, kw, ic, oc}, f32);
    auto outTy = RankedTensorType::get({b, oh, ow, oc}, f32);

    Value input =
        tensor::EmptyOp::create(builder, loc, inputTy.getShape(), f32);
    Value filter =
        tensor::EmptyOp::create(builder, loc, filterTy.getShape(), f32);
    Value out = tensor::EmptyOp::create(builder, loc, outTy.getShape(), f32);

    return linalg::Conv2DNhwcHwcfOp::create(builder, loc, TypeRange{outTy},
                                            ValueRange{input, filter},
                                            ValueRange{out});
  }

  // Pooling is a convolution interface op but has no outputChannel
  // or inputChannel dims, so inferContractionLikeDims should fail.
  linalg::PoolingNhwcSumOp getTestPoolingConv() {
    OpBuilder builder = setupBuilder();
    Location loc = builder.getUnknownLoc();
    auto f32 = builder.getF32Type();

    auto inputTy = RankedTensorType::get({1, 16, 16, 8}, f32);
    auto windowTy = RankedTensorType::get({3, 3}, f32);
    auto outTy = RankedTensorType::get({1, 14, 14, 8}, f32);

    Value input =
        tensor::EmptyOp::create(builder, loc, inputTy.getShape(), f32);
    Value window =
        tensor::EmptyOp::create(builder, loc, windowTy.getShape(), f32);
    Value out = tensor::EmptyOp::create(builder, loc, outTy.getShape(), f32);

    return linalg::PoolingNhwcSumOp::create(builder, loc, TypeRange{outTy},
                                            ValueRange{input, window},
                                            ValueRange{out});
  }

  func::FuncOp getTestFuncOp() {
    OpBuilder builder = setupBuilder();
    Location loc = builder.getUnknownLoc();

    auto f32 = builder.getF32Type();
    auto func = func::FuncOp::create(loc, "test",
                                     builder.getFunctionType({f32}, {f32}));

    builder.insert(func);
    return func;
  }

private:
  MLIRContext ctx;
  OwningOpRef<ModuleOp> module;
};

class RootOpUtilsTest : public LinalgTestBase {};

TEST_F(RootOpUtilsTest, InferContractionDimsForFillOp) {
  auto op = getTestFillOp();
  ASSERT_TRUE(!!op);
  EXPECT_TRUE(failed(inferContractionLikeDims(op)));
}

TEST_F(RootOpUtilsTest, InferContractionDimsForMatmulOp) {
  auto op = getTestMatmulOp();
  ASSERT_TRUE(!!op);

  auto dims = inferContractionLikeDims(op);

  ASSERT_TRUE(succeeded(dims));
  // matmul loops: (m=0, n=1, k=2).
  EXPECT_EQ(dims->m, SmallVector<unsigned>({0}));
  EXPECT_EQ(dims->n, SmallVector<unsigned>({1}));
  EXPECT_EQ(dims->k, SmallVector<unsigned>({2}));
}

TEST_F(RootOpUtilsTest, InferContractionDimsForConvOp) {
  auto op = getTestConv2DNhwcHwcfOp();
  ASSERT_TRUE(op);

  auto dims = inferContractionLikeDims(op);
  ASSERT_TRUE(succeeded(dims));

  // conv_2d_nhwc_hwcf loops: (b=0, oh=1, ow=2, oc=3, kh=4, kw=5, ic=6).
  // m = outputImage dims (oh, ow)
  // n = outputChannel dims(oc)
  // k = inputChannel dims (ic)
  EXPECT_EQ(dims->m, SmallVector<unsigned>({1, 2}));
  EXPECT_EQ(dims->n, SmallVector<unsigned>({3}));
  EXPECT_EQ(dims->k, SmallVector<unsigned>({6}));
}

TEST_F(RootOpUtilsTest, InferContractionDimsForEmptyDimConvOp) {
  auto op = getTestPoolingConv();
  ASSERT_TRUE(!!op);
  EXPECT_TRUE(failed(inferContractionLikeDims(op)));
}

TEST_F(RootOpUtilsTest, GetRootOpLoopInfoForMatmulOp) {
  unsigned int m = 16;
  unsigned int n = 16;
  unsigned int k = 32;
  auto op = getTestMatmulOp(m, n, k);
  ASSERT_TRUE(!!op);

  auto loopInfo = getRootOpLoopInfo(op);
  ASSERT_TRUE(loopInfo.has_value());
  EXPECT_EQ(loopInfo->numLoops, 3u);
  EXPECT_EQ(loopInfo->staticLoopRanges, SmallVector<int64_t>({m, n, k}));
  EXPECT_EQ(loopInfo->indexingMaps.size(), 3u);
}

TEST_F(RootOpUtilsTest, GetRootOpLoopInfoForConvOp) {
  auto op = getTestConv2DNhwcHwcfOp();
  ASSERT_TRUE(!!op);

  auto loopInfo = getRootOpLoopInfo(op);
  ASSERT_TRUE(loopInfo.has_value());
  // conv_2d_nhwc_hwcf: (b, oh, ow, oc, kh, kw, ic) = 7 loops.
  EXPECT_EQ(loopInfo->numLoops, 7u);
  EXPECT_EQ(loopInfo->indexingMaps.size(), 3u);
}

TEST_F(RootOpUtilsTest, GetRootOpLoopInfoForNonLinalgOp) {
  auto op = getTestFuncOp();
  ASSERT_TRUE(!!op);

  auto loopInfo = getRootOpLoopInfo(op);
  ASSERT_FALSE(loopInfo.has_value());
}

class CompatibleMMAAttrsTest : public LinalgTestBase {};

TEST_F(CompatibleMMAAttrsTest, F16InputF32AccOnRDNA4) {
  auto target = IREE::GPU::getHIPTargetDetails("gfx1201", "", getContext());
  ASSERT_TRUE(target);

  Type f16 = Float16Type::get(getContext());
  Type f32 = Float32Type::get(getContext());
  auto op = getTestMatmulOp(f16, f32);
  ASSERT_TRUE(!!op);

  auto loopInfo = getRootOpLoopInfo(op);
  ASSERT_TRUE(loopInfo.has_value());
  auto dims = inferContractionLikeDims(op);
  ASSERT_TRUE(succeeded(dims));
  auto result = getCompatibleMMAAttrs(op, target, *loopInfo, *dims);
  EXPECT_FALSE(result.empty());

  for (Attribute attr : result) {
    auto mma = dyn_cast<IREE::GPU::MmaInterfaceAttr>(attr);
    ASSERT_TRUE(!!mma);
    auto [aType, bType, cType] = mma.getABCElementTypes();
    EXPECT_EQ(aType, f16);
    EXPECT_EQ(bType, f16);
    EXPECT_EQ(cType, f32);
  }
}

TEST_F(CompatibleMMAAttrsTest, I8InputI32AccOnCDNA3) {
  auto target = IREE::GPU::getHIPTargetDetails("gfx942", "", getContext());
  ASSERT_TRUE(target);

  Type i8 = IntegerType::get(getContext(), 8);
  Type i32 = IntegerType::get(getContext(), 32);
  auto op = getTestMatmulOp(i8, i32);
  ASSERT_TRUE(!!op);

  auto loopInfo = getRootOpLoopInfo(op);
  ASSERT_TRUE(loopInfo.has_value());
  auto dims = inferContractionLikeDims(op);
  ASSERT_TRUE(succeeded(dims));
  auto result = getCompatibleMMAAttrs(op, target, *loopInfo, *dims);
  EXPECT_FALSE(result.empty());

  for (Attribute attr : result) {
    auto mma = dyn_cast<IREE::GPU::MmaInterfaceAttr>(attr);
    ASSERT_TRUE(!!mma);
    auto [aType, bType, cType] = mma.getABCElementTypes();
    EXPECT_EQ(aType, i8);
    EXPECT_EQ(bType, i8);
    EXPECT_EQ(cType, i32);
  }
}

TEST_F(CompatibleMMAAttrsTest, IncompatibleTypes) {
  auto target = IREE::GPU::getHIPTargetDetails("gfx942", "", getContext());
  ASSERT_TRUE(target);

  Type i1 = IntegerType::get(getContext(), 1);
  auto op = getTestMatmulOp(i1, i1);
  ASSERT_TRUE(!!op);

  auto loopInfo = getRootOpLoopInfo(op);
  auto dims = inferContractionLikeDims(op);
  auto result = getCompatibleMMAAttrs(op, target, *loopInfo, *dims);
  EXPECT_TRUE(result.empty());
}

/// Helper to build a knob variable name from a prefix and index.
/// e.g. ("wg_", 2) -> "wg_2".
static std::string makeVarName(StringRef prefix, unsigned idx) {
  return (prefix + Twine(idx)).str();
}

class VectorDistributeKnobsTest : public ::testing::Test {
protected:
  VectorDistributeKnobsTest() {
    DialectRegistry reg;
    reg.insert<IREE::Codegen::IREECodegenDialect>();
    ctx.appendDialectRegistry(reg);
    ctx.loadAllAvailableDialects();
  }

  MLIRContext *getContext() { return &ctx; }

  RootOpLoopInfo loopInfoForMatmul() {
    AffineExpr d0 = getAffineDimExpr(0, &ctx);
    AffineExpr d1 = getAffineDimExpr(1, &ctx);
    AffineExpr d2 = getAffineDimExpr(2, &ctx);
    return RootOpLoopInfo{
        /*staticLoopRanges=*/{16, 16, 32},
        /*numLoops=*/3,
        /*indexingMaps=*/
        {
            AffineMap::get(3, 0, {d0, d2}, &ctx), // LHS: (m, k)
            AffineMap::get(3, 0, {d2, d1}, &ctx), // RHS: (k, n)
            AffineMap::get(3, 0, {d0, d1}, &ctx), // Out: (m, n)
        }};
  }

  ContractionLikeDims matmulDims() {
    return ContractionLikeDims{/*m=*/{0}, /*n=*/{1}, /*k=*/{2}};
  }

  // conv_2d_nhwc_hwcf with input=1x18x18x4, filter=3x3x4x8, output=1x16x16x8.
  // 7 loops: (b, oh, ow, oc, kh, kw, ic).

  RootOpLoopInfo loopInfoForConv() {
    AffineExpr d0 = getAffineDimExpr(0, &ctx);
    AffineExpr d1 = getAffineDimExpr(1, &ctx);
    AffineExpr d2 = getAffineDimExpr(2, &ctx);
    AffineExpr d3 = getAffineDimExpr(3, &ctx);
    AffineExpr d4 = getAffineDimExpr(4, &ctx);
    AffineExpr d5 = getAffineDimExpr(5, &ctx);
    AffineExpr d6 = getAffineDimExpr(6, &ctx);
    return RootOpLoopInfo{
        /*staticLoopRanges=*/{1, 16, 16, 8, 3, 3, 4},
        /*numLoops=*/7,
        /*indexingMaps=*/
        {
            // Input: (b, oh+kh, ow+kw, ic)
            AffineMap::get(7, 0, {d0, d1 + d4, d2 + d5, d6}, &ctx),
            // Filter: (kh, kw, ic, oc)
            AffineMap::get(7, 0, {d4, d5, d6, d3}, &ctx),
            // Output: (b, oh, ow, oc)
            AffineMap::get(7, 0, {d0, d1, d2, d3}, &ctx),
        }};
  }

  ContractionLikeDims convDims() {
    return ContractionLikeDims{SmallVector<unsigned>({1, 2}),
                               SmallVector<unsigned>({3}),
                               SmallVector<unsigned>({6})};
  }

  SmallVector<Attribute> compatibleMMAs() {
    return SmallVector<Attribute>({
        StringAttr::get(&ctx, "mma_0"),
        StringAttr::get(&ctx, "mma_1"),
        StringAttr::get(&ctx, "mma_2"),
    });
  }

private:
  MLIRContext ctx;
};

TEST_F(VectorDistributeKnobsTest, KnobDictForMatmul) {
  DictionaryAttr dict = buildVectorDistributeKnobsDict(
      getContext(), loopInfoForMatmul(), matmulDims(), compatibleMMAs());
  ASSERT_TRUE(dict.get(kKnobWorkgroupKey));
  ASSERT_TRUE(dict.get(kKnobReductionKey));
  ASSERT_TRUE(dict.get(kKnobMmaKindKey));
  ASSERT_TRUE(dict.get(kKnobSubgroupBasisKey));
  ASSERT_TRUE(dict.get(kKnobWorkgroupSizeKey));
  ASSERT_TRUE(dict.get(kKnobSubgroupSizeKey));

  auto workgroup = cast<ArrayAttr>(dict.get(kKnobWorkgroupKey));
  ASSERT_EQ(workgroup.size(), 3u);
  EXPECT_EQ(cast<IntKnobAttr>(workgroup[0]).getName().getValue(),
            makeVarName(kKnobWgPrefix, 0));
  EXPECT_EQ(cast<IntKnobAttr>(workgroup[1]).getName().getValue(),
            makeVarName(kKnobWgPrefix, 1));
  EXPECT_EQ(cast<IntegerAttr>(workgroup[2]).getInt(), 0);

  auto reduction = cast<ArrayAttr>(dict.get(kKnobReductionKey));
  ASSERT_EQ(reduction.size(), 3u);
  EXPECT_EQ(cast<IntegerAttr>(reduction[0]).getInt(), 0);
  EXPECT_EQ(cast<IntegerAttr>(reduction[1]).getInt(), 0);
  EXPECT_EQ(cast<IntKnobAttr>(reduction[2]).getName().getValue(),
            makeVarName(kKnobRedPrefix, 2));

  auto mmaKind = cast<OneOfKnobAttr>(dict.get(kKnobMmaKindKey));
  EXPECT_EQ(mmaKind.getName().getValue(), kKnobMmaIdxName);
  EXPECT_EQ(mmaKind.getOptions().size(), 3u);

  auto sgBasis = cast<DictionaryAttr>(dict.get(kKnobSubgroupBasisKey));
  auto counts = cast<ArrayAttr>(sgBasis.get(kKnobCountsKey));
  ASSERT_EQ(counts.size(), 3u);
  EXPECT_EQ(cast<IntKnobAttr>(counts[0]).getName().getValue(), kKnobSgMCntName);
  EXPECT_EQ(cast<IntKnobAttr>(counts[1]).getName().getValue(), kKnobSgNCntName);
  EXPECT_EQ(cast<IntegerAttr>(counts[2]).getInt(), 1);
  // VectorDistribute matmul uses identity mapping.
  EXPECT_FALSE(sgBasis.get(kKnobMappingKey));

  auto wgSize = cast<ArrayAttr>(dict.get(kKnobWorkgroupSizeKey));
  ASSERT_EQ(wgSize.size(), 3u);
  EXPECT_EQ(cast<IntKnobAttr>(wgSize[0]).getName().getValue(),
            kKnobWgSizeXName);
  EXPECT_EQ(cast<IntKnobAttr>(wgSize[1]).getName().getValue(),
            kKnobWgSizeYName);
  EXPECT_EQ(cast<IntKnobAttr>(wgSize[2]).getName().getValue(),
            kKnobWgSizeZName);

  auto sgSize = cast<IntKnobAttr>(dict.get(kKnobSubgroupSizeKey));
  EXPECT_EQ(sgSize.getName().getValue(), kKnobSgSizeName);
}

TEST_F(VectorDistributeKnobsTest, KnobDictForConv) {
  DictionaryAttr dict = buildVectorDistributeKnobsDict(
      getContext(), loopInfoForConv(), convDims(), compatibleMMAs());
  ASSERT_TRUE(dict.get(kKnobWorkgroupKey));
  ASSERT_TRUE(dict.get(kKnobReductionKey));
  ASSERT_TRUE(dict.get(kKnobMmaKindKey));
  ASSERT_TRUE(dict.get(kKnobSubgroupBasisKey));
  ASSERT_TRUE(dict.get(kKnobWorkgroupSizeKey));
  ASSERT_TRUE(dict.get(kKnobSubgroupSizeKey));

  // 7 loops: (b, oh, ow, oc, kh, kw, ic). m={1,2}, n={3}, k={6}.
  auto workgroup = cast<ArrayAttr>(dict.get(kKnobWorkgroupKey));
  ASSERT_EQ(workgroup.size(), 7u);
  EXPECT_EQ(cast<IntegerAttr>(workgroup[0]).getInt(), 0);
  EXPECT_EQ(cast<IntKnobAttr>(workgroup[1]).getName().getValue(),
            makeVarName(kKnobWgPrefix, 1));
  EXPECT_EQ(cast<IntKnobAttr>(workgroup[2]).getName().getValue(),
            makeVarName(kKnobWgPrefix, 2));
  EXPECT_EQ(cast<IntKnobAttr>(workgroup[3]).getName().getValue(),
            makeVarName(kKnobWgPrefix, 3));
  for (unsigned i : {4u, 5u, 6u}) {
    EXPECT_EQ(cast<IntegerAttr>(workgroup[i]).getInt(), 0);
  }

  auto reduction = cast<ArrayAttr>(dict.get(kKnobReductionKey));
  ASSERT_EQ(reduction.size(), 7u);
  for (unsigned i = 0; i < 6; ++i) {
    EXPECT_EQ(cast<IntegerAttr>(reduction[i]).getInt(), 0);
  }
  EXPECT_EQ(cast<IntKnobAttr>(reduction[6]).getName().getValue(),
            makeVarName(kKnobRedPrefix, 6));

  auto mmaKind = cast<OneOfKnobAttr>(dict.get(kKnobMmaKindKey));
  EXPECT_EQ(mmaKind.getName().getValue(), kKnobMmaIdxName);
  EXPECT_EQ(mmaKind.getOptions().size(), 3u);

  auto sgBasis = cast<DictionaryAttr>(dict.get(kKnobSubgroupBasisKey));
  auto counts = cast<ArrayAttr>(sgBasis.get(kKnobCountsKey));
  ASSERT_EQ(counts.size(), 7u);
  for (unsigned i : {0u, 1u, 4u, 5u, 6u}) {
    EXPECT_EQ(cast<IntegerAttr>(counts[i]).getInt(), 1);
  }
  EXPECT_EQ(cast<IntKnobAttr>(counts[2]).getName().getValue(), kKnobSgMCntName);
  EXPECT_EQ(cast<IntKnobAttr>(counts[3]).getName().getValue(), kKnobSgNCntName);
  // VectorDistribute conv uses identity mapping.
  EXPECT_FALSE(sgBasis.get(kKnobMappingKey));

  auto wgSize = cast<ArrayAttr>(dict.get(kKnobWorkgroupSizeKey));
  ASSERT_EQ(wgSize.size(), 3u);
  EXPECT_EQ(cast<IntKnobAttr>(wgSize[0]).getName().getValue(),
            kKnobWgSizeXName);
  EXPECT_EQ(cast<IntKnobAttr>(wgSize[1]).getName().getValue(),
            kKnobWgSizeYName);
  EXPECT_EQ(cast<IntKnobAttr>(wgSize[2]).getName().getValue(),
            kKnobWgSizeZName);

  auto sgSize = cast<IntKnobAttr>(dict.get(kKnobSubgroupSizeKey));
  EXPECT_EQ(sgSize.getName().getValue(), kKnobSgSizeName);
}
} // namespace
} // namespace mlir::iree_compiler

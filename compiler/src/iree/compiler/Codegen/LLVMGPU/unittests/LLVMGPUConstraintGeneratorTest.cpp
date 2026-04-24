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
#include "mlir/Parser/Parser.h"

namespace mlir::iree_compiler {
namespace {

using IntKnobAttr = IREE::Codegen::IntKnobAttr;
using OneOfKnobAttr = IREE::Codegen::OneOfKnobAttr;

class RootOpUtilsTest : public ::testing::Test {
protected:
  RootOpUtilsTest() {
    DialectRegistry reg;
    reg.insert<arith::ArithDialect, func::FuncDialect, linalg::LinalgDialect,
               tensor::TensorDialect>();
    ctx.appendDialectRegistry(reg);
    ctx.loadAllAvailableDialects();
  }

  OpBuilder setupBuilder() {
    OpBuilder builder(&ctx);
    module = ModuleOp::create(builder.getUnknownLoc());
    builder.setInsertionPointToStart(module->getBody());
    return builder;
  }

  linalg::MatmulOp getTestMatmulOp(int64_t m = 16, int64_t n = 16,
                                   int64_t k = 32) {
    OpBuilder builder = setupBuilder();
    Location loc = builder.getUnknownLoc();

    auto f16 = builder.getF16Type();
    auto f32 = builder.getF32Type();

    auto lhsType = RankedTensorType::get({m, k}, f16);
    auto rhsType = RankedTensorType::get({k, n}, f16);
    auto accType = RankedTensorType::get({m, n}, f32);

    Value lhs = tensor::EmptyOp::create(builder, loc, lhsType.getShape(), f16);
    Value rhs = tensor::EmptyOp::create(builder, loc, rhsType.getShape(), f16);
    Value acc = tensor::EmptyOp::create(builder, loc, accType.getShape(), f32);

    return linalg::MatmulOp::create(builder, loc, TypeRange{accType},
                                    ValueRange{lhs, rhs}, ValueRange{acc});
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
  getTestConv2DNhwcHwcfOp(int64_t b = 1, int64_t ih = 3, int64_t iw = 3,
                          int64_t ic = 1, int64_t kh = 1, int64_t kw = 1,
                          int64_t oc = 1) {

    OpBuilder builder = setupBuilder();
    Location loc = builder.getUnknownLoc();
    auto f32 = builder.getF32Type();

    int64_t oh = ih - kh + 1;
    int64_t ow = iw - kw + 1;

    auto input_ty = RankedTensorType::get({b, ih, iw, ic}, f32);
    auto filter_ty = RankedTensorType::get({kh, kw, ic, oc}, f32);
    auto out_ty = RankedTensorType::get({b, oh, ow, oc}, f32);

    Value input =
        tensor::EmptyOp::create(builder, loc, input_ty.getShape(), f32);
    Value filter =
        tensor::EmptyOp::create(builder, loc, filter_ty.getShape(), f32);
    Value out = tensor::EmptyOp::create(builder, loc, out_ty.getShape(), f32);

    return linalg::Conv2DNhwcHwcfOp::create(builder, loc, TypeRange{out_ty},
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
    auto outTy = RankedTensorType::get({1, 14, 14, 8}, f32);

    Value input =
        tensor::EmptyOp::create(builder, loc, inputTy.getShape(), f32);
    Value out = tensor::EmptyOp::create(builder, loc, outTy.getShape(), f32);

    return linalg::PoolingNhwcSumOp::create(builder, loc, TypeRange{outTy},
                                            ValueRange{input}, ValueRange{out});
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

TEST_F(RootOpUtilsTest, InferContractionDims_FillFails) {
  auto op = getTestFillOp();
  ASSERT_TRUE(!!op);
  EXPECT_TRUE(failed(inferContractionLikeDims(op)));
}

TEST_F(RootOpUtilsTest, InferContractionDims_Matmul) {
  int64_t m = 32;
  int64_t n = 64;
  int64_t k = 8;
  auto op = getTestMatmulOp(m, n, k);
  ASSERT_TRUE(!!op);

  auto dims = inferContractionLikeDims(op);

  ASSERT_TRUE(succeeded(dims));
  EXPECT_EQ(dims->m, SmallVector<unsigned>({m}));
  EXPECT_EQ(dims->n, SmallVector<unsigned>({n}));
  EXPECT_EQ(dims->k, SmallVector<unsigned>({k}));
}

TEST_F(RootOpUtilsTest, InferContractionDims_Conv) {
  int64_t b = 1;
  int64_t ih = 5;
  int64_t iw = 6;
  int64_t ic = 3;
  int64_t kh = 2;
  int64_t kw = 3;
  int64_t oc = 8;

  auto op = getTestConv2DNhwcHwcfOp(b, ih, iw, ic, kh, kw, oc);
  ASSERT_TRUE(op);

  auto dims = inferContractionLikeDims(op);
  ASSERT_TRUE(succeeded(dims));

  int64_t oh = ih - kh + 1;
  int64_t ow = iw - kw + 1;

  // conv_2d_nhwc_hwcf:
  // m = output spatial dims (oh, ow)
  // n = output channel (oc)
  // k = reduction dims (kh, kw, ic)
  EXPECT_EQ(dims->m, SmallVector<unsigned>({oh, ow}));
  EXPECT_EQ(dims->n, SmallVector<unsigned>({oc}));
  EXPECT_EQ(dims->k, SmallVector<unsigned>({kh, kw, ic}));
}

TEST_F(RootOpUtilsTest, InferContractionDims_ConvEmptyDims) {
  auto op = getTestPoolingConv();
  ASSERT_TRUE(!!op);
  EXPECT_TRUE(failed(inferContractionLikeDims(op)));
}

TEST_F(RootOpUtilsTest, GetRootOpLoopInfo_Matmul) {
  int64_t m = 16;
  int64_t n = 16;
  int64_t k = 32;
  auto op = getTestMatmulOp(m, n, k);
  ASSERT_TRUE(!!op);

  auto info = getRootOpLoopInfo(op);
  ASSERT_TRUE(info.has_value());
  EXPECT_EQ(info->numLoops, 3u);
  EXPECT_EQ(info->staticLoopRanges, SmallVector<int64_t>({m, n, k}));
  EXPECT_EQ(info->indexingMaps.size(), 3u);
}

TEST_F(RootOpUtilsTest, GetRootOpLoopInfo_Conv) {
  auto op = getTestConv2DNhwcHwcfOp();
  ASSERT_TRUE(!!op);

  auto info = getRootOpLoopInfo(op);
  ASSERT_TRUE(info.has_value());
  // conv_2d_nhwc_hwcf: (b, oh, ow, oc, kh, kw, ic) = 7 loops.
  EXPECT_EQ(info->numLoops, 7u);
  EXPECT_EQ(info->indexingMaps.size(), 3u);
}

TEST_F(RootOpUtilsTest, GetRootOpLoopInfo_NonLinalg) {
  auto op = getTestFuncOp();
  ASSERT_TRUE(!!op);

  auto info = getRootOpLoopInfo(op);
  ASSERT_FALSE(info.has_value());
}

class CompatibleMMAAttrsTest : public ::testing::Test {
protected:
  CompatibleMMAAttrsTest() {
    DialectRegistry reg;
    reg.insert<IREE::GPU::IREEGPUDialect, arith::ArithDialect,
               func::FuncDialect, linalg::LinalgDialect,
               tensor::TensorDialect>();
    ctx.appendDialectRegistry(reg);
    ctx.loadAllAvailableDialects();
  }

  MLIRContext *getContext() { return &ctx; }

  IREE::GPU::TargetAttr getGfx942Target() {
    return IREE::GPU::getHIPTargetDetails("gfx942", "", &ctx);
  }

  IREE::GPU::TargetAttr getGfx1201Target() {
    return IREE::GPU::getHIPTargetDetails("gfx1201", "", &ctx);
  }

  linalg::MatmulOp getTestMatmulOp(Type inputType, Type accType, int64_t m = 16,
                                   int64_t n = 16, int64_t k = 32) {
    OpBuilder builder(&ctx);
    module = ModuleOp::create(builder.getUnknownLoc());
    builder.setInsertionPointToStart(module->getBody());
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

private:
  MLIRContext ctx;
  OwningOpRef<ModuleOp> module;
};

TEST_F(CompatibleMMAAttrsTest, F16InputF32AccOnRDNA4) {
  auto target = IREE::GPU::getHIPTargetDetails("gfx1201", "", getContext());
  ASSERT_TRUE(target);

  Type f16 = Float16Type::get(getContext());
  Type f32 = Float32Type::get(getContext());
  auto op = getTestMatmulOp(f16, f32);
  ASSERT_TRUE(!!op);

  auto loopInfo = getRootOpLoopInfo(op);
  auto dims = inferContractionLikeDims(op);
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

  auto loopInfo = getRootOpLoopInfo(op);
  auto dims = inferContractionLikeDims(op);
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

class VectorDistributeKnobsDictFixture : public ::testing::Test {
protected:
  VectorDistributeKnobsDictFixture() {
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

TEST_F(RootOpUtilsTest, BuildVectorDistributeKnobs_Matmul_Structure) {
  auto op = getTestMatmulOp();
  ASSERT_TRUE(!!op);

  auto info = getRootOpLoopInfo(op);
  ASSERT_TRUE(info.has_value());
  auto dims = inferContractionLikeDims(op);
  ASSERT_TRUE(succeeded(dims));

  auto compatibleMMAs = getCompatibleMMAAttrs(op, target, *info, *dims);

  DictionaryAttr dict = buildVectorDistributeKnobsDict(
      ctx, loopInfoForMatmul(), matmulDims(), compatibleMMAs());

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
  auto mapping = cast<ArrayAttr>(sgBasis.get(kKnobMappingKey));
  ASSERT_EQ(mapping.size(), 3u);

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

TEST_F(VectorDistributeKnobsDictFixture,
       BuildVectorDistributeKnobs_Conv_Structure) {
  MLIRContext *ctx = getContext();
  DictionaryAttr dict = buildVectorDistributeKnobsDict(
      ctx, loopInfoForConv(), convDims(), compatibleMMAs());

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
  auto mapping = cast<ArrayAttr>(sgBasis.get(kKnobMappingKey));
  ASSERT_EQ(mapping.size(), 7u);

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

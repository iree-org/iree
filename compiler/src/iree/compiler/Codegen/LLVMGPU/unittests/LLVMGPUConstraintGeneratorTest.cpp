// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <gtest/gtest.h>

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

class GetCompatibleMMAAttrsTest : public ::testing::Test {
protected:
  GetCompatibleMMAAttrsTest() {
    DialectRegistry reg;
    reg.insert<IREE::GPU::IREEGPUDialect>();
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

private:
  MLIRContext ctx;
};

TEST_F(GetCompatibleMMAAttrsTest, F16InputF32AccOnGfx942) {
  MLIRContext *ctx = getContext();
  auto target = getGfx942Target();
  ASSERT_TRUE(target);

  Type f16 = Float16Type::get(ctx);
  Type f32 = Float32Type::get(ctx);
  auto result = getCompatibleMMAAttrs(target, f16, f16, f32, ctx);
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

TEST_F(GetCompatibleMMAAttrsTest, IncludesVirtualMMA) {
  MLIRContext *ctx = getContext();
  auto target = getGfx942Target();
  ASSERT_TRUE(target);

  Type f16 = Float16Type::get(ctx);
  Type f32 = Float32Type::get(ctx);
  auto withoutVirtual = getCompatibleMMAAttrs(target, f16, f16, f32, ctx);
  auto withVirtual = getCompatibleMMAAttrs(target, f16, f16, f32, ctx,
                                           /*includeVirtual=*/true);
  EXPECT_GE(withVirtual.size(), withoutVirtual.size());
}

TEST_F(GetCompatibleMMAAttrsTest, I8InputI32AccOnRDNA4) {
  MLIRContext *ctx = getContext();
  auto target = getGfx1201Target();
  ASSERT_TRUE(target);

  Type i8 = IntegerType::get(ctx, 8);
  Type i32 = IntegerType::get(ctx, 32);
  auto result = getCompatibleMMAAttrs(target, i8, i8, i32, ctx);
  EXPECT_FALSE(result.empty());

  for (Attribute attr : result) {
    auto mma = dyn_cast<IREE::GPU::MmaInterfaceAttr>(attr);
    ASSERT_TRUE(!!mma);
    auto [aType, bType, cType] = mma.getABCElementTypes();
    EXPECT_EQ(aType, i8);
    EXPECT_EQ(bType, i8);
  }
}

TEST_F(GetCompatibleMMAAttrsTest, IncompatibleTypesReturnEmpty) {
  MLIRContext *ctx = getContext();
  auto target = getGfx942Target();
  ASSERT_TRUE(target);

  Type i1 = IntegerType::get(ctx, 1);
  auto result = getCompatibleMMAAttrs(target, i1, i1, i1, ctx);
  EXPECT_TRUE(result.empty());
}

TEST_F(GetCompatibleMMAAttrsTest, NoDuplicates) {
  MLIRContext *ctx = getContext();
  auto target = getGfx942Target();
  ASSERT_TRUE(target);

  Type f16 = Float16Type::get(ctx);
  Type f32 = Float32Type::get(ctx);
  auto result = getCompatibleMMAAttrs(target, f16, f16, f32, ctx,
                                      /*includeVirtual=*/true);
  for (size_t i = 0; i < result.size(); ++i) {
    for (size_t j = i + 1; j < result.size(); ++j) {
      EXPECT_NE(result[i], result[j]);
    }
  }
}

class LinalgOpFixture : public ::testing::Test {
protected:
  LinalgOpFixture() {
    DialectRegistry reg;
    reg.insert<arith::ArithDialect, func::FuncDialect, linalg::LinalgDialect,
               tensor::TensorDialect>();
    ctx.appendDialectRegistry(reg);
    ctx.loadAllAvailableDialects();
  }

  MLIRContext *getContext() { return &ctx; }

  linalg::LinalgOp parseLinalgOp(StringRef mlirText) {
    module = parseSourceString<ModuleOp>(mlirText, &ctx);
    linalg::LinalgOp result;
    module->walk([&](linalg::LinalgOp op) { result = op; });
    return result;
  }

  linalg::LinalgOp getFillOp() {
    return parseLinalgOp(R"(
      func.func @test(%val: f32, %out: tensor<16x16xf32>)
          -> tensor<16x16xf32> {
        %0 = linalg.fill ins(%val : f32) outs(%out : tensor<16x16xf32>)
            -> tensor<16x16xf32>
        return %0 : tensor<16x16xf32>
      }
    )");
  }

  linalg::LinalgOp getContractionOp() {
    return parseLinalgOp(R"(
      func.func @test(%lhs: tensor<16x32xf16>, %rhs: tensor<32x16xf16>,
                      %acc: tensor<16x16xf32>) -> tensor<16x16xf32> {
        %0 = linalg.matmul
            ins(%lhs, %rhs : tensor<16x32xf16>, tensor<32x16xf16>)
            outs(%acc : tensor<16x16xf32>) -> tensor<16x16xf32>
        return %0 : tensor<16x16xf32>
      }
    )");
  }

  linalg::LinalgOp getConvOp() {
    return parseLinalgOp(R"(
      func.func @test(%input: tensor<1x18x18x4xf32>,
                      %filter: tensor<3x3x4x8xf32>,
                      %output: tensor<1x16x16x8xf32>)
          -> tensor<1x16x16x8xf32> {
        %0 = linalg.conv_2d_nhwc_hwcf
            ins(%input, %filter
                : tensor<1x18x18x4xf32>, tensor<3x3x4x8xf32>)
            outs(%output : tensor<1x16x16x8xf32>)
            -> tensor<1x16x16x8xf32>
        return %0 : tensor<1x16x16x8xf32>
      }
    )");
  }

  // Pooling is a convolution interface op but has no outputChannel
  // or inputChannel dims, so inferContractionLikeDims should fail.
  linalg::LinalgOp getConvEmptyDimsOp() {
    return parseLinalgOp(R"(
      func.func @test(%input: tensor<1x4x4x1xf32>,
                      %kernel: tensor<3x3xf32>,
                      %output: tensor<1x2x2x1xf32>)
          -> tensor<1x2x2x1xf32> {
        %0 = linalg.pooling_nhwc_sum
            ins(%input, %kernel
                : tensor<1x4x4x1xf32>, tensor<3x3xf32>)
            outs(%output : tensor<1x2x2x1xf32>)
            -> tensor<1x2x2x1xf32>
        return %0 : tensor<1x2x2x1xf32>
      }
    )");
  }

private:
  MLIRContext ctx;
  OwningOpRef<ModuleOp> module;
};

TEST_F(LinalgOpFixture, InferContractionDims_FillFails) {
  auto op = getFillOp();
  ASSERT_TRUE(!!op);
  EXPECT_TRUE(failed(inferContractionLikeDims(op)));
}

TEST_F(LinalgOpFixture, InferContractionDims_Matmul) {
  auto op = getContractionOp();
  ASSERT_TRUE(!!op);

  auto dims = inferContractionLikeDims(op);
  ASSERT_TRUE(succeeded(dims));
  EXPECT_EQ(dims->m, SmallVector<unsigned>({0}));
  EXPECT_EQ(dims->n, SmallVector<unsigned>({1}));
  EXPECT_EQ(dims->k, SmallVector<unsigned>({2}));
}

TEST_F(LinalgOpFixture, InferContractionDims_Conv) {
  auto op = getConvOp();
  ASSERT_TRUE(!!op);

  auto dims = inferContractionLikeDims(op);
  ASSERT_TRUE(succeeded(dims));
  // conv_2d_nhwc_hwcf: outputImage={1,2}, outputChannel={3},
  // inputChannel={6} (mapped to m, n, k).
  EXPECT_EQ(dims->m, SmallVector<unsigned>({1, 2}));
  EXPECT_EQ(dims->n, SmallVector<unsigned>({3}));
  EXPECT_EQ(dims->k, SmallVector<unsigned>({6}));
}

TEST_F(LinalgOpFixture, InferContractionDims_ConvEmptyDimsFails) {
  auto op = getConvEmptyDimsOp();
  ASSERT_TRUE(!!op);
  EXPECT_TRUE(failed(inferContractionLikeDims(op)));
}

TEST_F(LinalgOpFixture, GetRootOpLoopInfo_Matmul) {
  auto op = getContractionOp();
  ASSERT_TRUE(!!op);

  auto info = getRootOpLoopInfo(op);
  ASSERT_TRUE(info.has_value());
  EXPECT_EQ(info->numLoops, 3u);
  EXPECT_EQ(info->staticLoopRanges, SmallVector<int64_t>({16, 16, 32}));
  EXPECT_EQ(info->indexingMaps.size(), 3u);
}

TEST_F(LinalgOpFixture, GetRootOpLoopInfo_Conv) {
  auto op = getConvOp();
  ASSERT_TRUE(!!op);

  auto info = getRootOpLoopInfo(op);
  ASSERT_TRUE(info.has_value());
  // conv_2d_nhwc_hwcf: (n, oh, ow, oc, kh, kw, ic) = 7 loops.
  EXPECT_EQ(info->numLoops, 7u);
  EXPECT_EQ(info->indexingMaps.size(), 3u);
}

TEST_F(LinalgOpFixture, GetRootOpLoopInfo_NonLinalgFails) {
  auto module = parseSourceString<ModuleOp>(R"(
    func.func @test(%x: f32) -> f32 { return %x : f32 }
  )",
                                            getContext());
  ASSERT_TRUE(!!module);

  func::FuncOp func;
  module->walk([&](func::FuncOp f) { func = f; });
  ASSERT_TRUE(!!func);
  EXPECT_FALSE(getRootOpLoopInfo(func).has_value());
}

class BuildVectorDistributeKnobsDictTest : public ::testing::Test {
protected:
  BuildVectorDistributeKnobsDictTest() {
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
  // 7 loops: (n, oh, ow, oc, kh, kw, ic).
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
            // Input: (n, oh+kh, ow+kw, ic)
            AffineMap::get(7, 0, {d0, d1 + d4, d2 + d5, d6}, &ctx),
            // Filter: (kh, kw, ic, oc)
            AffineMap::get(7, 0, {d4, d5, d6, d3}, &ctx),
            // Output: (n, oh, ow, oc)
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

TEST_F(BuildVectorDistributeKnobsDictTest,
       BuildVectorDistributeKnobs_Matmul_DictStr) {
  MLIRContext *ctx = getContext();
  DictionaryAttr matmulKnobDict = buildVectorDistributeKnobsDict(
      ctx, loopInfoForMatmul(), matmulDims(), compatibleMMAs());

  std::string result;
  llvm::raw_string_ostream os(result);
  matmulKnobDict.print(os);

  StringRef expected =
      "{"
      "mma_kind = #iree_codegen.smt.one_of_knob<\"mma_idx\", "
      "[\"mma_0\", \"mma_1\", \"mma_2\"]>, "
      "reduction = [0, 0, #iree_codegen.smt.int_knob<\"red_2\">], "
      "subgroup_basis = {"
      "counts = [#iree_codegen.smt.int_knob<\"sg_m_cnt\">, "
      "#iree_codegen.smt.int_knob<\"sg_n_cnt\">, 1], "
      "mapping = [0, 1, 2]}, "
      "subgroup_size = #iree_codegen.smt.int_knob<\"sg_size\">, "
      "workgroup = [#iree_codegen.smt.int_knob<\"wg_0\">, "
      "#iree_codegen.smt.int_knob<\"wg_1\">, 0], "
      "workgroup_size = [#iree_codegen.smt.int_knob<\"wg_x\">, "
      "#iree_codegen.smt.int_knob<\"wg_y\">, "
      "#iree_codegen.smt.int_knob<\"wg_z\">]"
      "}";
  EXPECT_EQ(result, expected);
}

TEST_F(BuildVectorDistributeKnobsDictTest,
       BuildVectorDistributeKnobs_Conv_DictStr) {
  MLIRContext *ctx = getContext();
  DictionaryAttr convKnobDict = buildVectorDistributeKnobsDict(
      ctx, loopInfoForConv(), convDims(), compatibleMMAs());

  std::string result;
  llvm::raw_string_ostream os(result);
  convKnobDict.print(os);

  StringRef expected =
      "{"
      "mma_kind = #iree_codegen.smt.one_of_knob<\"mma_idx\", "
      "[\"mma_0\", \"mma_1\", \"mma_2\"]>, "
      "reduction = [0, 0, 0, 0, 0, 0, #iree_codegen.smt.int_knob<\"red_6\">], "
      "subgroup_basis = {"
      "counts = [1, 1, "
      "#iree_codegen.smt.int_knob<\"sg_m_cnt\">, "
      "#iree_codegen.smt.int_knob<\"sg_n_cnt\">, "
      "1, 1, 1], "
      "mapping = [0, 1, 2, 3, 4, 5, 6]}, "
      "subgroup_size = #iree_codegen.smt.int_knob<\"sg_size\">, "
      "workgroup = [0, "
      "#iree_codegen.smt.int_knob<\"wg_1\">, "
      "#iree_codegen.smt.int_knob<\"wg_2\">, "
      "#iree_codegen.smt.int_knob<\"wg_3\">, "
      "0, 0, 0], "
      "workgroup_size = [#iree_codegen.smt.int_knob<\"wg_x\">, "
      "#iree_codegen.smt.int_knob<\"wg_y\">, "
      "#iree_codegen.smt.int_knob<\"wg_z\">]"
      "}";
  EXPECT_EQ(result, expected);
}
} // namespace
} // namespace mlir::iree_compiler

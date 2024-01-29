// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Common/GPU/PassDetail.h"
#include "iree/compiler/Codegen/Common/GPU/Passes.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"

namespace mlir::iree_compiler {

/// This function implements the following swizzling logic
/// void getTiledId2(unsigned x, unsigned y, unsigned* tiledx,
///                 unsigned* tiledy) {
///  unsigned t_tiledx = (x + (y % tile) * grid_size_x) / tile;
///  unsigned t_tiledy = (y / tile) * tile +
///      (x + (y % tile) * grid_size_x) % tile;
///  bool c = grid_size_y % tile != 0 &&
///      ((y / tile) * tile + tile) > grid_size_y;
///  *tiledx = c ? x : t_tiledx;
///  *tiledy = c ? y : t_tiledy;
/// }
// TODO: Make this a callback and the core functionality in the pass a utility
// function.
static void makeSwizzledId(Location loc, OpBuilder b, Value workgroupIdX,
                           Value workgroupIdY, Value gridSizeX, Value gridSizeY,
                           Value &SwizzledIdX, Value &SwizzledIdY,
                           unsigned swizzleTile) {
  Value zero = b.create<arith::ConstantIndexOp>(loc, 0);
  Value tile = b.create<arith::ConstantIndexOp>(loc, swizzleTile);
  Value yModTile = b.create<arith::RemUIOp>(loc, workgroupIdY, tile);
  Value yDivTile = b.create<arith::DivUIOp>(loc, workgroupIdY, tile);
  Value swizzleParam = b.create<arith::MulIOp>(loc, yModTile, gridSizeX);
  Value swizzleParam2 =
      b.create<arith::AddIOp>(loc, workgroupIdX, swizzleParam);
  Value swizzleParam3 = b.create<arith::RemUIOp>(loc, swizzleParam2, tile);
  Value swizzleParam4 = b.create<arith::MulIOp>(loc, yDivTile, tile);
  Value unboundedSwizzledIdX =
      b.create<arith::DivUIOp>(loc, swizzleParam2, tile);
  Value unboundedSwizzledIdY =
      b.create<arith::AddIOp>(loc, swizzleParam3, swizzleParam4);
  Value gyModTile = b.create<arith::RemUIOp>(loc, gridSizeY, tile);
  Value gyAddTile = b.create<arith::AddIOp>(loc, swizzleParam4, tile);
  Value condition1 =
      b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ne, gyModTile, zero);
  Value condition2 = b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ugt,
                                             gyAddTile, gridSizeY);
  Value condition3 = b.create<arith::AndIOp>(loc, condition1, condition2);
  SwizzledIdX = b.create<arith::SelectOp>(loc, condition3, workgroupIdX,
                                          unboundedSwizzledIdX);
  SwizzledIdY = b.create<arith::SelectOp>(loc, condition3, workgroupIdY,
                                          unboundedSwizzledIdY);
}

LogicalResult swizzleWorkgroupsInFunc(mlir::FunctionOpInterface funcOp,
                                      unsigned swizzleLogTile) {
  if (swizzleLogTile == 0)
    return success();
  unsigned swizzleTile = pow(2, swizzleLogTile);
  std::array<IREE::HAL::InterfaceWorkgroupIDOp, 2> oldWorkgroupIds;
  bool xFound = false, yFound = false;
  funcOp.walk([&](IREE::HAL::InterfaceWorkgroupIDOp idOp) {
    unsigned index = idOp.getDimension().getZExtValue();
    if (index == 0) {
      oldWorkgroupIds[index] = idOp;
      xFound = true;
    } else if (index == 1) {
      oldWorkgroupIds[index] = idOp;
      yFound = true;
    }
  });
  if (xFound == false || yFound == false)
    return failure();
  OpBuilder builder(funcOp);
  builder.setInsertionPoint(&funcOp.front(), funcOp.front().begin());
  Value workgroupIdX =
      builder.create<IREE::HAL::InterfaceWorkgroupIDOp>(funcOp.getLoc(), 0);
  Value workgroupIdY =
      builder.create<IREE::HAL::InterfaceWorkgroupIDOp>(funcOp.getLoc(), 1);
  Value gridSizeX =
      builder.create<IREE::HAL::InterfaceWorkgroupCountOp>(funcOp.getLoc(), 0);
  Value gridSizeY =
      builder.create<IREE::HAL::InterfaceWorkgroupCountOp>(funcOp.getLoc(), 1);
  Value SwizzledIdX, SwizzledIdY;
  makeSwizzledId(funcOp.getLoc(), builder, workgroupIdX, workgroupIdY,
                 gridSizeX, gridSizeY, SwizzledIdX, SwizzledIdY, swizzleTile);
  oldWorkgroupIds[0].replaceAllUsesWith(SwizzledIdX);
  oldWorkgroupIds[1].replaceAllUsesWith(SwizzledIdY);
  return success();
}

namespace {
struct WorkGroupSwizzlePass
    : public WorkGroupSwizzleBase<WorkGroupSwizzlePass> {
  WorkGroupSwizzlePass(unsigned swizzleLogTile)
      : swizzleLogTile(swizzleLogTile) {}

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<affine::AffineDialect>();
  }
  LogicalResult initializeOptions(StringRef options) override {
    if (failed(Pass::initializeOptions(options))) {
      return failure();
    }
    swizzleLogTile = logTile;
    return success();
  }
  void runOnOperation() override {
    auto funcOp = getOperation();
    (void)swizzleWorkgroupsInFunc(funcOp, swizzleLogTile);
  }

private:
  unsigned swizzleLogTile;
};
} // namespace

std::unique_ptr<InterfacePass<mlir::FunctionOpInterface>>
createWorkGroupSwizzle(unsigned swizzleLogTile) {
  return std::make_unique<WorkGroupSwizzlePass>(swizzleLogTile);
}

} // namespace mlir::iree_compiler

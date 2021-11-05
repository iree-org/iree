// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/PassDetail.h"
#include "iree/compiler/Codegen/Passes.h"
#include "llvm/Support/FormatVariadic.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Pass/Pass.h"

namespace mlir {
namespace iree_compiler {

namespace {

// void fn(int k, void* lhs, void* rhs, void* dst)
struct MMT4DExternSpec {
  std::string name;
  FunctionType funcType;
  IntegerType kType;
  ShapedType lhsType;
  ShapedType rhsType;
  ShapedType dstType;
};

static StringRef getTypeLiteral(Type type) {
  if (auto intType = type.dyn_cast<IntegerType>()) {
    switch (intType.getIntOrFloatBitWidth()) {
      case 8:
        return "i8";
      case 16:
        return "i16";
      case 32:
        return "i32";
      case 64:
        return "i64";
      default:
        llvm_unreachable("unhandled MMT4D int type");
        break;
    }
  } else if (auto floatType = type.dyn_cast<FloatType>()) {
    switch (floatType.getIntOrFloatBitWidth()) {
      case 8:
        return "f8";
      case 16:
        return "f16";
      case 32:
        return "f32";
      case 64:
        return "f64";
      default:
        llvm_unreachable("unhandled MMT4D float type");
        break;
    }
  } else {
    llvm_unreachable("unhandled MMT4D type");
  }
}

// Returns true if the given m0/k0/n0 are supported for externalization.
static bool isSupported(int64_t m0, int64_t k0, int64_t n0, Type lhsType,
                        Type rhsType, Type dstType) {
  return m0 == 8 && k0 == 4 && n0 == 8 && lhsType.isInteger(8) &&
         rhsType.isInteger(8) && dstType.isInteger(32);
}

static llvm::Optional<MMT4DExternSpec> makeExternSpec(linalg::Mmt4DOp mmt4DOp,
                                                      Value lhs, Value rhs,
                                                      Value dst) {
  MMT4DExternSpec externSpec;
  externSpec.kType = IntegerType::get(mmt4DOp.getContext(), 32);

  // linalg.mmt4d
  //     ins(%lhs, %rhs: memref<M1xK1xM0xK0xf32>, memref<N1xK1xN0xK0xf32>)
  //     outs(%dst: memref<M1xN1xM0xN0xf32>)
  auto lhsRawType = lhs.getType().cast<ShapedType>();
  auto rhsRawType = rhs.getType().cast<ShapedType>();
  auto dstRawType = dst.getType().cast<ShapedType>();
  auto lhsElementType = lhsRawType.getElementType();
  auto rhsElementType = rhsRawType.getElementType();
  auto dstElementType = dstRawType.getElementType();

  int64_t m0 = lhsRawType.getDimSize(2);
  int64_t k0 = lhsRawType.getDimSize(3);
  int64_t n0 = rhsRawType.getDimSize(2);
  if (!isSupported(m0, k0, n0, lhsElementType, rhsElementType,
                   dstElementType)) {
    return llvm::None;
  }

  externSpec.lhsType = MemRefType::get(
      {-1, -1, lhsRawType.getDimSize(2), lhsRawType.getDimSize(3)},
      lhsRawType.getElementType());
  externSpec.rhsType = MemRefType::get(
      {-1, -1, rhsRawType.getDimSize(2), rhsRawType.getDimSize(3)},
      rhsRawType.getElementType());
  externSpec.dstType = MemRefType::get(
      {-1, -1, dstRawType.getDimSize(2), dstRawType.getDimSize(3)},
      dstRawType.getElementType());

  externSpec.funcType = FunctionType::get(mmt4DOp.getContext(),
                                          TypeRange{
                                              externSpec.kType,
                                              externSpec.lhsType,
                                              externSpec.rhsType,
                                              externSpec.dstType,
                                          },
                                          TypeRange{});

  // NOTE: we could look at the MMT4D op and pull off any attributes that may
  // exist to change which function we dispatch.
  externSpec.name = llvm::formatv(
      "mmt4d_{0}x{1}x{2}_{3}{4}{5}", m0, k0, n0, getTypeLiteral(lhsElementType),
      getTypeLiteral(rhsElementType), getTypeLiteral(dstElementType));

  return externSpec;
}

struct LLVMCPUExternalizeMMT4DPass
    : public LLVMCPUExternalizeMMT4DBase<LLVMCPUExternalizeMMT4DPass> {
  LLVMCPUExternalizeMMT4DPass() = default;

  void runOnOperation() override {
    auto moduleOp = getOperation();

    DenseSet<StringRef> externSet;
    SmallVector<MMT4DExternSpec> externSpecs;
    for (auto funcOp : moduleOp.getOps<FuncOp>()) {
      funcOp.walk([&](linalg::Mmt4DOp mmt4DOp) {
        OpBuilder builder(mmt4DOp);
        auto loc = mmt4DOp.getLoc();

        auto lhsRaw = mmt4DOp.getInputOperand(0)->get();
        auto rhsRaw = mmt4DOp.getInputOperand(1)->get();
        auto dstRaw = mmt4DOp.getOutputOperand(0)->get();

        auto externSpecOr = makeExternSpec(mmt4DOp, lhsRaw, rhsRaw, dstRaw);
        if (!externSpecOr.hasValue()) return;
        auto externSpec = externSpecOr.getValue();
        if (externSet.insert(externSpec.name).second) {
          externSpecs.push_back(externSpec);
        }

        // linalg.mmt4d
        //     ins(%lhs, %rhs: memref<M1xK1xM0xK0xf32>, memref<N1xK1xN0xK0xf32>)
        //     outs(%dst: memref<M1xN1xM0xN0xf32>)
        auto k0 = builder.createOrFold<memref::DimOp>(loc, lhsRaw, 3);
        auto k1 = builder.createOrFold<memref::DimOp>(loc, lhsRaw, 1);
        auto k = builder.createOrFold<arith::MulIOp>(loc, k0, k1);
        auto kInt =
            builder.createOrFold<arith::IndexCastOp>(loc, externSpec.kType, k);

        auto lhsDynamic = builder.createOrFold<memref::CastOp>(
            loc, externSpec.lhsType, lhsRaw);
        auto rhsDynamic = builder.createOrFold<memref::CastOp>(
            loc, externSpec.rhsType, rhsRaw);
        auto dstDynamic = builder.createOrFold<memref::CastOp>(
            loc, externSpec.dstType, dstRaw);

        builder.create<mlir::CallOp>(
            loc, externSpec.name, TypeRange{},
            ValueRange{kInt, lhsDynamic, rhsDynamic, dstDynamic});

        mmt4DOp.erase();
      });
    }

    auto moduleBuilder = OpBuilder::atBlockBegin(&moduleOp.body().front());
    for (auto externSpec : externSpecs) {
      auto externOp = moduleBuilder.create<mlir::FuncOp>(
          moduleOp.getLoc(), externSpec.name, externSpec.funcType);
      externOp.setPrivate();
    }
  }
};

}  // namespace

std::unique_ptr<OperationPass<ModuleOp>> createLLVMCPUExternalizeMMT4DPass() {
  return std::make_unique<LLVMCPUExternalizeMMT4DPass>();
}

}  // namespace iree_compiler
}  // namespace mlir

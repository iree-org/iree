// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <utility>

#include "iree/compiler/Dialect/Flow/IR/FlowDialect.h"
#include "iree/compiler/Dialect/Stream/IR/StreamDialect.h"
#include "iree/compiler/Dialect/Stream/IR/StreamOps.h"
#include "iree/compiler/Dialect/Stream/IR/StreamTypes.h"
#include "iree/compiler/Dialect/Stream/Transforms/PassDetail.h"
#include "iree/compiler/Dialect/Stream/Transforms/Passes.h"
#include "iree/compiler/Dialect/Util/IR/UtilDialect.h"
#include "iree/compiler/Dialect/Util/IR/UtilOps.h"
#include "llvm/ADT/TypeSwitch.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/Pass/Pass.h"

// We could overengineer this with custom DSLs and python generation and all
// that kind of stuff - but that stuff really belongs closer to the frontend
// (linalg/etc). Once we are at this point we are just patching over things for
// compatibility and not handling arbitrary programs. A linalg.fill of an i64
// that gets tiled and fused with other operations is always going to be several
// of orders of magnitude faster than this approach and we should spend our
// effort improving things at that layer instead of leaning too much on this.
//
// Consider these as replacements for the blobs we'd have to ship with every
// deployment of the runtime (for all target platforms/HAL backends/etc) - in
// that sense this is a dramatically more scalable approach even if not perfect.

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace Stream {

// To ensure deterministic insertion order of executables we use std::map.
// We have < ~10 builtins so it's not a very big set.
using BuiltinUsageMap =
    std::map<StringRef, SmallVector<IREE::Stream::BuiltinOpInterface>>;

// Returns all builtins used in the module.
static BuiltinUsageMap findBuiltinOps(mlir::ModuleOp moduleOp) {
  BuiltinUsageMap results;
  for (auto callableOp : moduleOp.getOps<CallableOpInterface>()) {
    auto *region = callableOp.getCallableRegion();
    if (!region) continue;
    for (auto &block : *region) {
      for (auto &op : block.getOperations()) {
        if (auto builtinOp = dyn_cast<IREE::Stream::BuiltinOpInterface>(op)) {
          auto name = builtinOp->getName();
          results[name.getStringRef()].push_back(builtinOp);
        }
      }
    }
  }
  return results;
}

class MaterializeBuiltinsPass
    : public MaterializeBuiltinsBase<MaterializeBuiltinsPass> {
 public:
  MaterializeBuiltinsPass() = default;

  void getDependentDialects(DialectRegistry &registry) const override {
    // We need to include all dialects that the builtin modules use.
    registry.insert<mlir::func::FuncDialect>();
    registry.insert<mlir::arith::ArithDialect>();
    registry.insert<mlir::linalg::LinalgDialect>();
    registry.insert<mlir::memref::MemRefDialect>();
    registry.insert<mlir::scf::SCFDialect>();
    registry.insert<mlir::vector::VectorDialect>();
    registry.insert<IREE::Flow::FlowDialect>();
    registry.insert<IREE::Stream::StreamDialect>();
    registry.insert<IREE::Util::UtilDialect>();
  }

  void runOnOperation() override {
    auto moduleOp = getOperation();
    if (moduleOp.getBody()->empty()) return;

    // Find all builtin ops used throughout the module.
    // We only want to materialize each executable once and may want to
    // specialize it based on usage.
    auto builtinUsageMap = findBuiltinOps(moduleOp);

    // Materialize each builtin type.
    OpBuilder moduleBuilder(&moduleOp.getBody()->front());
    for (auto it : builtinUsageMap) {
      // Merge the builtin module contents into our target module.
      // We only want to do this once per builtin type.
      auto anyOp = it.second.front();
      if (failed(anyOp.mergeBuiltinModule(moduleOp, moduleBuilder))) {
        return signalPassFailure();
      }

      // Replace each builtin op with the logic to dispatch it.
      for (auto op : it.second) {
        OpBuilder builder(op);
        if (failed(op.convertBuiltinOp(builder))) {
          return signalPassFailure();
        }
        op.erase();
      }
    }
  }
};

std::unique_ptr<OperationPass<mlir::ModuleOp>> createMaterializeBuiltinsPass() {
  return std::make_unique<MaterializeBuiltinsPass>();
}

}  // namespace Stream
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir

// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <memory>
#include <utility>

#include "iree/compiler/Dialect/HAL/IR/HALDialect.h"
#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
#include "iree/compiler/Dialect/HAL/Transforms/Passes.h"
#include "iree/compiler/Dialect/Stream/IR/StreamTypes.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Pass/Pass.h"

namespace mlir::iree_compiler::IREE::HAL {

#define GEN_PASS_DEF_RESOLVEDEVICEPROMISESPASS
#include "iree/compiler/Dialect/HAL/Transforms/Passes.h.inc"

namespace {

//===----------------------------------------------------------------------===//
// --iree-hal-resolve-device-promises
//===----------------------------------------------------------------------===//

struct ResolveDevicePromisesPass
    : public IREE::HAL::impl::ResolveDevicePromisesPassBase<
          ResolveDevicePromisesPass> {
  using IREE::HAL::impl::ResolveDevicePromisesPassBase<
      ResolveDevicePromisesPass>::ResolveDevicePromisesPassBase;

  void runOnOperation() override {
    auto moduleOp = getOperation();

    // Resolves a #hal.device.promise attr to a #hal.device.affinity. Fails if
    // the referenced device is not found.
    SymbolTable symbolTable(moduleOp);
    auto resolvePromise = [&](Operation *fromOp,
                              IREE::HAL::DevicePromiseAttr promiseAttr)
        -> FailureOr<IREE::Stream::AffinityAttr> {
      auto deviceOp =
          symbolTable.lookupNearestSymbolFrom<IREE::Util::GlobalOpInterface>(
              fromOp, promiseAttr.getDevice());
      if (!deviceOp) {
        return fromOp->emitOpError()
               << "references a promised device that was not declared: "
               << promiseAttr;
      }
      return cast<IREE::Stream::AffinityAttr>(
          IREE::HAL::DeviceAffinityAttr::get(&getContext(),
                                             FlatSymbolRefAttr::get(deviceOp),
                                             promiseAttr.getQueueMask()));
    };

    // Resolves any #hal.device.promise attr on the op.
    auto resolvePromiseAttrs = [&](Operation *op, DictionaryAttr attrDict)
        -> std::optional<std::pair<DictionaryAttr, WalkResult>> {
      bool didReplaceAny = false;
      auto newDict = dyn_cast_if_present<DictionaryAttr>(attrDict.replace(
          [&](Attribute attr)
              -> std::optional<std::pair<Attribute, WalkResult>> {
            if (auto promiseAttr =
                    dyn_cast_if_present<IREE::HAL::DevicePromiseAttr>(attr)) {
              auto resolvedAttrOr = resolvePromise(op, promiseAttr);
              if (failed(resolvedAttrOr)) {
                return std::make_pair(attr, WalkResult::interrupt());
              }
              didReplaceAny = true;
              return std::make_pair(resolvedAttrOr.value(),
                                    WalkResult::advance());
            }
            return std::nullopt;
          }));
      if (newDict) {
        return std::make_pair(newDict, didReplaceAny ? WalkResult::advance()
                                                     : WalkResult::skip());
      } else {
        return std::make_pair(attrDict, WalkResult::interrupt());
      }
    };
    auto resolveAllPromiseAttrs =
        [&](Operation *op,
            MutableArrayRef<DictionaryAttr> attrDicts) -> WalkResult {
      bool didReplaceAny = false;
      for (auto &attrDict : attrDicts) {
        auto resolveState = resolvePromiseAttrs(op, attrDict);
        if (!resolveState) {
          // Failed to resolve while recursively replacing.
          return WalkResult::interrupt();
        } else if (!resolveState->second.wasSkipped()) {
          // Performed a replacement.
          attrDict = resolveState->first;
          didReplaceAny = true;
        }
      }
      return didReplaceAny ? WalkResult::advance() : WalkResult::skip();
    };
    auto resolvePromisesOnOp = [&](Operation *op) -> WalkResult {
      auto opAttrs = op->getAttrDictionary();
      if (opAttrs) {
        auto resolveState = resolvePromiseAttrs(op, opAttrs);
        if (!resolveState) {
          // Failed to resolve while recursively replacing.
          return WalkResult::interrupt();
        } else if (!resolveState->second.wasSkipped()) {
          // Performed a replacement.
          op->setAttrs(resolveState->first);
        }
      }
      if (auto funcOp = dyn_cast<FunctionOpInterface>(op)) {
        SmallVector<DictionaryAttr> argAttrs;
        funcOp.getAllArgAttrs(argAttrs);
        auto argStatus = resolveAllPromiseAttrs(op, argAttrs);
        if (argStatus.wasInterrupted()) {
          return argStatus;
        } else if (!argStatus.wasSkipped()) {
          funcOp.setAllArgAttrs(argAttrs);
        }
        SmallVector<DictionaryAttr> resultAttrs;
        funcOp.getAllResultAttrs(resultAttrs);
        auto resultStatus = resolveAllPromiseAttrs(op, resultAttrs);
        if (resultStatus.wasInterrupted()) {
          return resultStatus;
        } else if (!resultStatus.wasSkipped()) {
          funcOp.setAllResultAttrs(resultAttrs);
        }
      }
      return WalkResult::advance();
    };

    // Walk the entire module and replace promises.
    // We skip any symbol table op as all devices are top-level only.
    if (resolvePromisesOnOp(moduleOp).wasInterrupted()) {
      return signalPassFailure();
    }
    if (moduleOp
            .walk([&](Operation *op) {
              if (op->hasTrait<OpTrait::SymbolTable>()) {
                return WalkResult::skip(); // ignore isolated ops
              }
              return resolvePromisesOnOp(op);
            })
            .wasInterrupted()) {
      return signalPassFailure();
    }
  }
};

} // namespace

} // namespace mlir::iree_compiler::IREE::HAL

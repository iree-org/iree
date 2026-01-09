// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/Stream/Transforms/Utils.h"

#include "iree/compiler/Dialect/Encoding/IR/EncodingTypes.h"
#include "iree/compiler/Dialect/Encoding/Utils/Utils.h"
#include "iree/compiler/Dialect/Stream/IR/StreamOps.h"
#include "iree/compiler/Dialect/Util/IR/UtilTypes.h"
#include "llvm/Support/Debug.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"

#define DEBUG_TYPE "iree-stream-transforms-utils"

namespace mlir::iree_compiler::IREE::Stream {

SmallVector<Attribute> getBindingLayoutAttrs(TensorDispatchOp dispatchOp) {
  SmallVector<int64_t> tiedOperands(dispatchOp.getNumResults(),
                                    IREE::Util::TiedOpInterface::kUntiedIndex);
  if (std::optional<ArrayAttr> tiedOperandsAttr =
          dispatchOp.getTiedOperands()) {
    tiedOperands =
        llvm::map_to_vector(tiedOperandsAttr.value(), [](Attribute intAttr) {
          return cast<IntegerAttr>(intAttr).getInt();
        });
  }

  SmallVector<Attribute> result(dispatchOp.getOperandEncodings().getValue());
  for (auto [resultEncoding, tiedOperand] : llvm::zip_equal(
           dispatchOp.getResultEncodings().getValue(), tiedOperands)) {
    if (tiedOperand != IREE::Util::TiedOpInterface::kUntiedIndex) {
      continue;
    }
    result.push_back(resultEncoding);
  }

  return result;
}

bool recognizeDispatchEntryPoints(ModuleOp moduleOp, SymbolTable &symbolTable,
                                  TensorDispatchOp dispatchOp) {
  bool result = true;
  dispatchOp.forEachEntryPointAttr([&](SymbolRefAttr entryPoint) {
    if (!result) {
      return;
    }
    auto exportOp = dyn_cast_if_present<ExecutableExportOp>(
        symbolTable.lookupSymbolIn(moduleOp, entryPoint));
    if (!exportOp) {
      result = false;
      return;
    }
    auto executableOp = exportOp->getParentOfType<ExecutableOp>();
    if (!executableOp) {
      result = false;
      return;
    }

    auto funcOp = cast<mlir::FunctionOpInterface>(symbolTable.lookupSymbolIn(
        executableOp.getInnerModule(), exportOp.getSymName()));
    for (auto arg : funcOp.getArguments()) {
      if (!isa<BindingType>(arg.getType())) {
        continue;
      }
      for (auto user : arg.getUsers()) {
        auto subspanOp = dyn_cast<BindingSubspanOp>(user);
        if (!subspanOp) {
          result = false;
          return;
        }
        auto encodingTypeInterface =
            dyn_cast<IREE::Encoding::EncodingTypeInterface>(
                subspanOp.getType());
        if (!encodingTypeInterface) {
          result = false;
          return;
        }
      }
    }
  });
  return result;
}

LogicalResult
updateBindingEncodings(FunctionOpInterface funcOp,
                       ArrayRef<Attribute> bindingLayoutTypeAttrs) {
  Region &region = funcOp.getFunctionBody();
  for (auto [arg, newTypeAttr] :
       llvm::zip_equal(region.getArguments(), bindingLayoutTypeAttrs)) {
    if (!isa<BindingType>(arg.getType())) {
      continue;
    }
    auto newType =
        dyn_cast<RankedTensorType>(cast<TypeAttr>(newTypeAttr).getValue());
    if (!newType) {
      LLVM_DEBUG(llvm::dbgs()
                 << "Skip, the new type is not RankedTensorType.\n");
      continue;
    }
    auto encodingAttr = IREE::Encoding::getSerializableAttr(newType);
    if (!encodingAttr) {
      LLVM_DEBUG(llvm::dbgs() << "Skip, the binding layout attribute is not "
                                 "SerializableAttr, which means that the type "
                                 "does not have a valid encoding.\n");
      continue;
    }
    for (auto user : arg.getUsers()) {
      auto subspanOp = cast<BindingSubspanOp>(user);
      auto encodingTypeInterface =
          cast<IREE::Encoding::EncodingTypeInterface>(subspanOp.getType());
      subspanOp.getResult().setType(
          encodingTypeInterface.updateEncoding(encodingAttr));
    }
  }
  return success();
}

LogicalResult
duplicateExecutablesPerLayoutVariant(ModuleOp moduleOp,
                                     SymbolTable &symbolTable,
                                     ArrayRef<TensorDispatchOp> candidates) {
  MLIRContext *ctx = moduleOp.getContext();
  IRRewriter rewriter(ctx);

  //===--------------------------------------------------------------------===//
  // Gather per-export [binding layouts] map. A function in an executable can be
  // run with different affinities. The function arguments, where the types are
  // `!stream.binding`, are consumed by `stream.binding.subspan` ops, and the op
  // returns a tensor type. The binding layouts indicate the resolved layouts
  // for those tensor types. The map records the mapping between an export op
  // and the possible binding layouts.
  //===--------------------------------------------------------------------===//
  DenseMap<ExecutableExportOp, SetVector<ArrayAttr>>
      bindingLayoutSetPerExportOp;

  // Records the binding layouts for a dispatch op.
  llvm::MapVector<TensorDispatchOp, SmallVector<Attribute>>
      dispatchOpBindingLayouts;
  for (auto dispatchOp : candidates) {
    SmallVector<Attribute> bindingLayoutAttrs =
        getBindingLayoutAttrs(dispatchOp);
    dispatchOpBindingLayouts[dispatchOp] = bindingLayoutAttrs;
    dispatchOp.forEachEntryPointAttr([&](SymbolRefAttr entryPoint) {
      auto exportOp = cast<ExecutableExportOp>(
          symbolTable.lookupSymbolIn(moduleOp, entryPoint));
      bindingLayoutSetPerExportOp[exportOp].insert(
          rewriter.getArrayAttr(bindingLayoutAttrs));
    });
  }

  LLVM_DEBUG({
    llvm::dbgs() << "Dump of bindingLayoutSetPerExportOp\n";
    for (auto [exportOp, layoutSet] : bindingLayoutSetPerExportOp) {
      llvm::dbgs() << "  ExportOp: " << exportOp.getSymName() << "\n";
      for (auto [idx, attr] : llvm::enumerate(layoutSet)) {
        llvm::dbgs() << "    binding_layouts #" << idx << ": " << attr << "\n ";
      }
    }
  });

  //===--------------------------------------------------------------------===//
  // Duplicate executables for each unique binding layouts.
  //===--------------------------------------------------------------------===//
  // Mapping from [export op, binding layouts] to the executable op. So we can
  // use it to update dispatch sites later on.
  using ExportAndBindingLayouts = std::pair<ExecutableExportOp, ArrayAttr>;
  DenseMap<ExportAndBindingLayouts, ExecutableOp> dispatchSiteToExecutableOp;
  for (auto [exportOp, layoutSet] : bindingLayoutSetPerExportOp) {
    int64_t dupId = -1;
    auto executableOp = exportOp->getParentOfType<ExecutableOp>();
    for (ArrayAttr bindingLayoutTypeAttrs : layoutSet) {
      rewriter.setInsertionPointAfter(executableOp);
      ExecutableOp dupOp = executableOp;
      if (dupId != -1) {
        auto symName = std::string(executableOp.getSymName());
        symName += "_dup" + std::to_string(dupId);
        dupOp = rewriter.cloneWithoutRegions(executableOp);
        rewriter.modifyOpInPlace(dupOp, [&] {
          dupOp.setSymName(symName);
          IRMapping mapping;
          executableOp.getRegion().cloneInto(&dupOp.getRegion(), mapping);
        });
      }

      // Update the binding encodings within the cloned executable op.
      auto funcOp = cast<mlir::FunctionOpInterface>(symbolTable.lookupSymbolIn(
          dupOp.getInnerModule(), exportOp.getSymName()));
      if (failed(updateBindingEncodings(funcOp,
                                        bindingLayoutTypeAttrs.getValue()))) {
        return funcOp->emitOpError("failed to update encodings for bindings");
      }
      dispatchSiteToExecutableOp[ExportAndBindingLayouts(
          exportOp, bindingLayoutTypeAttrs)] = dupOp;
      dupId++;
    }
  }

  //===--------------------------------------------------------------------===//
  // Update dispatch sites, i.e., point dispatch entry points to corresponding
  // duplicated executables.
  //===--------------------------------------------------------------------===//
  for (auto dispatchOp : candidates) {
    SmallVector<Attribute> newEntryPoints;
    SmallVector<Attribute> bindingLayoutAttrs =
        dispatchOpBindingLayouts[dispatchOp];
    dispatchOp.forEachEntryPointAttr([&](SymbolRefAttr entryPoint) {
      auto exportOp = cast<ExecutableExportOp>(
          symbolTable.lookupSymbolIn(moduleOp, entryPoint));
      auto info = ExportAndBindingLayouts(
          exportOp, rewriter.getArrayAttr(bindingLayoutAttrs));
      assert(dispatchSiteToExecutableOp.count(info));

      auto executableOp = dispatchSiteToExecutableOp[info];
      auto newSym = SymbolRefAttr::get(executableOp->getAttrOfType<StringAttr>(
                                           SymbolTable::getSymbolAttrName()),
                                       entryPoint.getNestedReferences());
      newEntryPoints.push_back(newSym);
    });

    rewriter.modifyOpInPlace(dispatchOp, [&] {
      dispatchOp.setEntryPointsAttr(rewriter.getArrayAttr(newEntryPoints));
    });
  }
  return success();
}

} // namespace mlir::iree_compiler::IREE::Stream

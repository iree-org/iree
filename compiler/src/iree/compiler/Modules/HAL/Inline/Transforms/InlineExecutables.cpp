// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/HAL/IR/HALDialect.h"
#include "iree/compiler/Dialect/Stream/IR/StreamDialect.h"
#include "iree/compiler/Dialect/Stream/IR/StreamOps.h"
#include "iree/compiler/Dialect/Util/IR/UtilDialect.h"
#include "iree/compiler/Modules/HAL/Inline/IR/HALInlineDialect.h"
#include "iree/compiler/Modules/HAL/Inline/Transforms/PassDetail.h"
#include "iree/compiler/Modules/HAL/Inline/Transforms/Passes.h"
#include "iree/compiler/Utils/IndexSet.h"
#include "iree/compiler/Utils/ModuleUtils.h"
#include "llvm/ADT/STLExtras.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"

namespace mlir::iree_compiler::IREE::HAL {
namespace Inline {

class InlineExecutablesPass
    : public InlineExecutablesBase<InlineExecutablesPass> {
public:
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<IREE::Util::UtilDialect, IREE::HAL::HALDialect,
                    IREE::HAL::Inline::HALInlineDialect, arith::ArithDialect,
                    func::FuncDialect, scf::SCFDialect>();
  }

  void runOnOperation() override {
    auto moduleOp = getOperation();

    // Inline variants and produce a function map.
    DenseMap<Attribute, Attribute> exportToFuncMap;
    SymbolTableCollection symbolTables;
    for (auto executableOp : llvm::make_early_inc_range(
             moduleOp.getOps<IREE::HAL::ExecutableOp>())) {
      // Inline each variant.
      for (auto variantOp :
           executableOp.getOps<IREE::HAL::ExecutableVariantOp>()) {
        if (failed(inlineVariant(executableOp, variantOp, moduleOp,
                                 exportToFuncMap, symbolTables))) {
          return signalPassFailure();
        }
      }

      // Drop executable after information has been extracted and the workgroup
      // code has been inlined.
      executableOp.erase();
    }

    // Annotate all dispatches with the target function.
    for (auto funcOp : moduleOp.getOps<mlir::FunctionOpInterface>()) {
      auto result = funcOp.walk([&](IREE::Stream::CmdDispatchOp dispatchOp) {
        // Specify new target function that conversion can use to make the call.
        // We only support single variant dispatches when inline.
        auto entryPointAttrs = dispatchOp.getEntryPoints().getValue();
        if (entryPointAttrs.size() != 1) {
          dispatchOp.emitOpError()
              << "multiple variant targets not supported with the inline HAL";
          return WalkResult::interrupt();
        }
        auto targetFuncName =
            llvm::cast<StringAttr>(exportToFuncMap[entryPointAttrs.front()]);
        assert(targetFuncName && "missing mapping");
        dispatchOp->setAttr("hal_inline.target",
                            FlatSymbolRefAttr::get(targetFuncName));
        return WalkResult::advance();
      });
      if (result.wasInterrupted()) {
        return signalPassFailure();
      }
    }
  }

  LogicalResult inlineVariant(IREE::HAL::ExecutableOp executableOp,
                              IREE::HAL::ExecutableVariantOp variantOp,
                              mlir::ModuleOp targetModuleOp,
                              DenseMap<Attribute, Attribute> &exportToFuncMap,
                              SymbolTableCollection &symbolTables) {
    auto innerModuleOp = variantOp.getInnerModule();
    auto innerSymbolTable = symbolTables.getSymbolTable(innerModuleOp);
    auto innerModuleBuilder = OpBuilder::atBlockEnd(innerModuleOp.getBody());

    // We want to merge the module ahead of the exported functions to ensure
    // initializer order is preserved.
    OpBuilder targetModuleBuilder(executableOp);

    // Build each dispatch function wrapper.
    auto indexType = innerModuleBuilder.getIndexType();
    auto i32Type = innerModuleBuilder.getI32Type();
    auto bufferType = innerModuleBuilder.getType<IREE::Util::BufferType>();
    for (auto exportOp : variantOp.getExportOps()) {
      // Build dispatch function signature that the stream.cmd.dispatch ops will
      // map to.
      auto layoutAttr = exportOp.getLayout();
      size_t totalBindingCount = 0;
      for (auto setLayout : layoutAttr.getSetLayouts()) {
        totalBindingCount += setLayout.getBindings().size();
      }
      SmallVector<Type> inputTypes;
      inputTypes.append(exportOp.getWorkgroupCountBody()->getNumArguments() - 1,
                        indexType); // workload
      inputTypes.append(layoutAttr.getPushConstants(), i32Type);
      inputTypes.append(totalBindingCount, bufferType); // buffers
      inputTypes.append(totalBindingCount, indexType);  // offsets
      inputTypes.append(totalBindingCount, indexType);  // lengths
      auto dispatchFuncType =
          innerModuleBuilder.getFunctionType(inputTypes, {});

      // Create the function and insert into the module.
      auto dispatchFuncOp = func::FuncOp::create(
          exportOp.getLoc(),
          ("__dispatch_" + executableOp.getName() + "_" + exportOp.getName())
              .str(),
          dispatchFuncType);
      dispatchFuncOp.setPrivate();
      innerSymbolTable.insert(dispatchFuncOp,
                              innerModuleBuilder.getInsertionPoint());
      innerModuleBuilder.setInsertionPointAfter(dispatchFuncOp);

      // Build the dispatch function by calling the target function in a loop.
      auto bodyFuncOp =
          innerSymbolTable.lookup<func::FuncOp>(exportOp.getName());
      if (!bodyFuncOp) {
        return exportOp.emitOpError("missing body function");
      }
      if (bodyFuncOp.isPublic()) {
        if (failed(rewriteWorkgroupSignature(layoutAttr, totalBindingCount,
                                             bodyFuncOp))) {
          return failure();
        }
        bodyFuncOp.setPrivate(); // so we only do it once
      }
      buildDispatchFunc(exportOp, layoutAttr, totalBindingCount, bodyFuncOp,
                        dispatchFuncOp);

      // Map from what the stream.cmd.dispatch ops is using to the new function.
      auto exportTargetAttr = SymbolRefAttr::get(
          executableOp.getNameAttr(),
          {
              FlatSymbolRefAttr::get(variantOp.getNameAttr()),
              FlatSymbolRefAttr::get(exportOp.getNameAttr()),
          });
      exportToFuncMap[exportTargetAttr] = dispatchFuncOp.getNameAttr();
    }

    // Merge the source executable module into the target host module.
    if (failed(mergeModuleInto(innerModuleOp, targetModuleOp,
                               targetModuleBuilder))) {
      return failure();
    }

    return success();
  }

  // Rewrites a workgroup body function signature to a flattened list.
  //
  // Body (as translated):
  //   (local_memory, [constants], [bindings],
  //    workgroup_x, workgroup_y, workgroup_z,
  //    workgroup_size_x, workgroup_size_y, workgroup_size_z,
  //    workgroup_count_x, workgroup_count_y, workgroup_count_z)
  //
  // Body after rewrite:
  //   (local_memory, constants..., bindings...,
  //    workgroup_x, workgroup_y, workgroup_z,
  //    workgroup_size_x, workgroup_size_y, workgroup_size_z,
  //    workgroup_count_x, workgroup_count_y, workgroup_count_z)
  //
  // To make this process easier and lighten the load on the downstream passes
  // we muck with the ABI to pass a flattened list of constants and bindings.
  // Whenever better IPO and util.list optimizations are added we could back
  // this out to keep things vanilla and have fewer places making assumptions
  // about the function signatures.
  LogicalResult
  rewriteWorkgroupSignature(IREE::HAL::PipelineLayoutAttr layoutAttr,
                            size_t totalBindingCount, func::FuncOp bodyFuncOp) {
    auto *entryBlock = &bodyFuncOp.front();
    auto builder = OpBuilder::atBlockBegin(entryBlock);
    auto indexType = builder.getIndexType();
    auto i32Type = builder.getI32Type();
    auto bufferType = builder.getType<IREE::Util::BufferType>();

    // There may be nicer ways of doing this but I can't find them.
    // We build a new list of argument types and insert them as we go. This lets
    // us map the arguments over and replace usage such that by the end we can
    // slice off the original arguments as they'll have no more uses.
    unsigned originalArgCount = entryBlock->getNumArguments();
    SmallVector<Type> newArgTypes;
    unsigned argOffset = 0;

    // Local memory is carried across as-is.
    auto localMemoryArg = entryBlock->getArgument(argOffset++);
    newArgTypes.push_back(bufferType);
    localMemoryArg.replaceAllUsesWith(
        entryBlock->addArgument(bufferType, localMemoryArg.getLoc()));

    // Expand push constants by replacing buffer accesses with the flattened
    // args.
    newArgTypes.append(layoutAttr.getPushConstants(), i32Type);
    auto constantBuffer = entryBlock->getArgument(argOffset++);
    SmallVector<Value> constantArgs;
    for (unsigned i = 0; i < layoutAttr.getPushConstants(); ++i) {
      constantArgs.push_back(
          entryBlock->addArgument(i32Type, constantBuffer.getLoc()));
    }
    if (failed(replaceBufferAccesses(constantBuffer, constantArgs))) {
      return failure();
    }

    // Expand buffer list by replacing list accesses with the flattened args.
    newArgTypes.append(totalBindingCount, bufferType);
    auto bindingList = entryBlock->getArgument(argOffset++);
    SmallVector<Value> bindingArgs;
    for (unsigned i = 0; i < totalBindingCount; ++i) {
      bindingArgs.push_back(
          entryBlock->addArgument(bufferType, bindingList.getLoc()));
    }
    if (failed(replaceListAccesses(bindingList, bindingArgs))) {
      return failure();
    }

    // Take care of the workgroup id/size/count tuples.
    auto entryBuilder = OpBuilder::atBlockBegin(entryBlock);
    for (unsigned i = 0; i < 3 * /*xyz=*/3; ++i) {
      newArgTypes.push_back(indexType);
      auto oldArg = entryBlock->getArgument(argOffset++);
      auto newArg = entryBlock->addArgument(indexType, oldArg.getLoc());
      oldArg.replaceAllUsesWith(entryBuilder.create<arith::IndexCastOp>(
          oldArg.getLoc(), i32Type, newArg));
    }

    // Erase the original args.
    for (unsigned i = 0; i < originalArgCount; ++i) {
      entryBlock->eraseArgument(0);
    }

    // Update function signature to reflect the entry block args.
    bodyFuncOp.setType(
        builder.getFunctionType(newArgTypes, bodyFuncOp.getResultTypes()));

    return success();
  }

  // Replaces trivial constant index accesses to a buffer with their values.
  // This is an extremely poor optimization that we should remove if buffer
  // ever gets store-load forwarding - we could just create the buffer, store
  // the elements, and let that take care of the rest. Today it doesn't do that.
  LogicalResult replaceBufferAccesses(Value buffer, ValueRange elements) {
    for (auto user : llvm::make_early_inc_range(buffer.getUsers())) {
      if (auto sizeOp = dyn_cast<IREE::Util::BufferSizeOp>(user)) {
        // Ignored but we need to get rid of it.
        // TODO(benvanik): see if we can allow this through; today it will pin
        // the function argument (constants most likely) and cause us to fail to
        // remove it later on.
        Value dummySize = OpBuilder(sizeOp).create<arith::ConstantIndexOp>(
            sizeOp.getLoc(), 0xCAFEF00D);
        sizeOp.replaceAllUsesWith(dummySize);
        sizeOp.erase();
        continue;
      } else if (auto loadOp = dyn_cast<IREE::Util::BufferLoadOp>(user)) {
        APInt index;
        if (matchPattern(loadOp.getSourceOffset(), m_ConstantInt(&index))) {
          loadOp.replaceAllUsesWith(
              elements[index.getSExtValue() / sizeof(uint32_t)]);
          loadOp.erase();
          continue;
        } else {
          return loadOp.emitOpError(
              "unhandled dynamic buffer access; must be static");
        }
      } else if (auto loadOp = dyn_cast<memref::LoadOp>(user)) {
        if (loadOp.getIndices().size() != 1) {
          return loadOp.emitOpError(
              "expected memrefs to have been flattened before inlining "
              "executables");
        }
        APInt index;
        if (matchPattern(loadOp.getIndices()[0], m_ConstantInt(&index))) {
          loadOp.replaceAllUsesWith(elements[index.getSExtValue()]);
          loadOp.erase();
          continue;
        } else {
          return loadOp.emitOpError(
              "unhandled dynamic buffer access; must be static");
        }
      } else {
        return user->emitOpError(
            "unhandled buffer access op; only loads are supported");
      }
    }
    return success();
  }

  // Replaces trivial constant index accesses to a list with their values.
  // util.list store-load forwarding could do this instead.
  LogicalResult replaceListAccesses(Value list, ValueRange elements) {
    for (auto user : llvm::make_early_inc_range(list.getUsers())) {
      if (auto getOp = dyn_cast<IREE::Util::ListGetOp>(user)) {
        APInt index;
        if (matchPattern(getOp.getIndex(), m_ConstantInt(&index))) {
          getOp.replaceAllUsesWith(elements[index.getSExtValue()]);
          getOp.erase();
          continue;
        } else {
          return getOp.emitOpError(
              "unhandled dynamic list access; must be static");
        }
      } else {
        return user->emitOpError(
            "unhandled list access op; only gets are supported");
      }
    }
    return success();
  }

  // Builds a function that calls a workgroup body and marshals arguments.
  //
  // Incoming:
  //   (workload..., push_constants...,
  //    binding_buffers..., binding_offsets..., binding_lengths...)
  // Body (as translated):
  //   (local_memory, [constants], [bindings],
  //    workgroup_x, workgroup_y, workgroup_z,
  //    workgroup_size_x, workgroup_size_y, workgroup_size_z,
  //    workgroup_count_x, workgroup_count_y, workgroup_count_z)
  void buildDispatchFunc(IREE::HAL::ExecutableExportOp exportOp,
                         IREE::HAL::PipelineLayoutAttr layoutAttr,
                         size_t totalBindingCount, func::FuncOp bodyFuncOp,
                         func::FuncOp dispatchFuncOp) {
    auto loc = exportOp.getLoc();
    auto builder = OpBuilder::atBlockBegin(dispatchFuncOp.addEntryBlock());
    IndexSet indexSet(loc, builder);
    auto bufferType = builder.getType<IREE::Util::BufferType>();

    SmallVector<Value> workgroupArgs;

    // Calculate the XYZ workgroup count from the export function.
    // There may be multiple exports pointing at the same body with different
    // workgroup count functions.
    unsigned workloadArgCount =
        exportOp.getWorkgroupCountBody()->getNumArguments() - 1;
    unsigned argOffset = 0;
    SmallVector<Value> workload;
    workload.reserve(workloadArgCount);
    for (unsigned i = 0; i < workloadArgCount; ++i) {
      workload.push_back(dispatchFuncOp.getArgument(argOffset++));
    }
    Value device = builder.create<IREE::Util::NullOp>(
        loc, builder.getType<IREE::HAL::DeviceType>());
    auto workgroupCount =
        exportOp.calculateWorkgroupCount(loc, device, workload, builder);

    // For now we don't handle local memory.
    Value localMemory = builder.create<IREE::Util::NullOp>(loc, bufferType);
    workgroupArgs.push_back(localMemory);

    // Pass all constants through.
    for (int64_t i = 0; i < layoutAttr.getPushConstants(); ++i) {
      workgroupArgs.push_back(dispatchFuncOp.getArgument(argOffset++));
    }

    // Pass all buffers through as subspans with the binding offset and length
    // factored in. IPO can propagate the subspans (hopefully).
    for (size_t i = 0; i < totalBindingCount; ++i) {
      auto bindingBuffer = dispatchFuncOp.getArgument(argOffset + i);
      auto bindingOffset =
          dispatchFuncOp.getArgument(argOffset + totalBindingCount + i);
      auto bindingLength = dispatchFuncOp.getArgument(
          argOffset + totalBindingCount + totalBindingCount + i);
      Value bufferSize =
          builder.create<IREE::Util::BufferSizeOp>(loc, bindingBuffer);
      Value bindingView = builder.create<IREE::Util::BufferSubspanOp>(
          loc, bindingBuffer, bufferSize, bindingOffset, bindingLength);
      workgroupArgs.push_back(bindingView);
    }

    int workgroupXYZOffset = workgroupArgs.size();
    workgroupArgs.push_back(nullptr);           // workgroup_x, set below
    workgroupArgs.push_back(nullptr);           // workgroup_y, set below
    workgroupArgs.push_back(nullptr);           // workgroup_z, set below
    workgroupArgs.append(3, indexSet.get(1));   // workgroup_size_xyz
    workgroupArgs.push_back(workgroupCount[0]); // workgroup_count_x
    workgroupArgs.push_back(workgroupCount[1]); // workgroup_count_y
    workgroupArgs.push_back(workgroupCount[2]); // workgroup_count_z

    // Z -> Y -> Z loop nest.
    builder.create<scf::ForOp>(
        loc, indexSet.get(0), workgroupCount[2], indexSet.get(1), ValueRange{},
        [&](OpBuilder &forZBuilder, Location loc, Value iz, ValueRange iters) {
          workgroupArgs[workgroupXYZOffset + 2] = iz;
          forZBuilder.create<scf::ForOp>(
              loc, indexSet.get(0), workgroupCount[1], indexSet.get(1),
              ValueRange{},
              [&](OpBuilder &forYBuilder, Location loc, Value iy,
                  ValueRange iters) {
                workgroupArgs[workgroupXYZOffset + 1] = iy;
                forYBuilder.create<scf::ForOp>(
                    loc, indexSet.get(0), workgroupCount[0], indexSet.get(1),
                    ValueRange{},
                    [&](OpBuilder &forXBuilder, Location loc, Value ix,
                        ValueRange iters) {
                      workgroupArgs[workgroupXYZOffset + 0] = ix;
                      forXBuilder.create<func::CallOp>(loc, bodyFuncOp,
                                                       workgroupArgs);
                      forXBuilder.create<scf::YieldOp>(loc);
                    });
                forYBuilder.create<scf::YieldOp>(loc);
              });
          forZBuilder.create<scf::YieldOp>(loc);
        });

    builder.create<func::ReturnOp>(loc);
  }
};

std::unique_ptr<OperationPass<mlir::ModuleOp>> createInlineExecutablesPass() {
  return std::make_unique<InlineExecutablesPass>();
}

} // namespace Inline
} // namespace mlir::iree_compiler::IREE::HAL

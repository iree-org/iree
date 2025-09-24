// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <memory>
#include <utility>

#include "../../compiler/plugins/target/EXSLERATEV2/exsleratev2/Transforms/MetaData.h"
#include "iree/compiler/Dialect/HAL/Analysis/DeviceAnalysis.h"
#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
#include "iree/compiler/Dialect/HAL/Target/TargetRegistry.h"
#include "iree/compiler/Dialect/HAL/Transforms/Passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/IR/LinalgInterfaces.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/Pass/Pass.h"

namespace mlir::iree_compiler::IREE::HAL {

#define GEN_PASS_DEF_VERIFYDEVICESPASS
#include "iree/compiler/Dialect/HAL/Transforms/Passes.h.inc"

namespace {

//===----------------------------------------------------------------------===//
// --iree-hal-verify-devices
//===----------------------------------------------------------------------===//


static void printAvailable(InFlightDiagnostic &diagnostic,
                           const TargetRegistry &targetRegistry) {
  diagnostic << "available devices: [";
  llvm::interleaveComma(targetRegistry.getRegisteredTargetDevices(),
                        diagnostic);
  diagnostic << "], available backends = [";
  llvm::interleaveComma(targetRegistry.getRegisteredTargetBackends(),
                        diagnostic);
  diagnostic << "]";
}

static LogicalResult
verifyDeviceTargetAttr(Operation *deviceOp,
                       IREE::HAL::DeviceTargetAttr deviceTargetAttr,
                       const TargetRegistry &targetRegistry) {
  auto targetDevice =
      targetRegistry.getTargetDevice(deviceTargetAttr.getDeviceID().getValue());
  if (!targetDevice) {
    auto diagnostic = deviceOp->emitError();
    diagnostic << "unregistered target device "
               << deviceTargetAttr.getDeviceID()
               << "; ensure it is linked into the compiler (available = [ ";
    for (const auto &targetName : targetRegistry.getRegisteredTargetDevices()) {
      diagnostic << "'" << targetName << "' ";
    }
    diagnostic << "])";
    return diagnostic;
  }

  for (auto executableTargetAttr : deviceTargetAttr.getExecutableTargets()) {
    auto targetBackend = targetRegistry.getTargetBackend(
        executableTargetAttr.getBackend().getValue());
    if (!targetBackend) {
      auto diagnostic = deviceOp->emitError();
      diagnostic << "unregistered target backend "
                 << executableTargetAttr.getBackend()
                 << "; ensure it is linked into the compiler (available = [ ";
      for (const auto &targetName :
           targetRegistry.getRegisteredTargetBackends()) {
        diagnostic << "'" << targetName << "' ";
      }
      diagnostic << "])";
      return diagnostic;
    }
  }

  return success();
}

static LogicalResult verifyAttr(Operation *deviceOp, Attribute attr,
                                const TargetRegistry &targetRegistry) {
  return TypeSwitch<Attribute, LogicalResult>(attr)
      .Case<IREE::HAL::DeviceTargetAttr>([&](auto deviceTargetAttr) {
        return verifyDeviceTargetAttr(deviceOp, deviceTargetAttr,
                                      targetRegistry);
      })
      .Case<IREE::HAL::DeviceSelectAttr>([&](auto deviceSelectAttr) {
        for (auto attr : deviceSelectAttr.getDevices().getValue()) {
          if (failed(verifyAttr(deviceOp, attr, targetRegistry))) {
            return failure();
          }
        }
        return success();
      })
      .Default([&](auto attr) {
        return success(); // probably fallback/ordinal/etc - can't verify
      });
}

struct VerifyDevicesPass
    : public IREE::HAL::impl::VerifyDevicesPassBase<VerifyDevicesPass> {
  using IREE::HAL::impl::VerifyDevicesPassBase<
      VerifyDevicesPass>::VerifyDevicesPassBase;
  void runOnOperation() override {
    mlir::ModuleOp moduleOp = getOperation();

    // Devices are required if we need to convert host code or executables.
    // If we only have hal.executables as input then we can bypass this.
    // We could extend this check to be a bit smarter at the risk of false
    // negatives - today this is just handling the standalone hal.executable
    // compilation workflow.
    bool anyNonExecutableOps = false;
    for (auto &op : moduleOp.getOps()) {
      if (!isa<IREE::HAL::ExecutableOp>(op)) {
        anyNonExecutableOps = true;
        break;
      }
    }
    if (!anyNonExecutableOps) {
      return;
    }

    // Analyze the module to find all devices.
    DeviceAnalysis deviceAnalysis(moduleOp);
    if (failed(deviceAnalysis.run())) {
      return signalPassFailure();
    }

    // Devices are only required if we have dialects we may lower into device
    // code. For now checking for tensor types is probably sufficient though we
    // may want a pluggable way to decide this (e.g. dialect/type/op
    // interfaces).
    auto isTensor = [](Type type) { return isa<TensorType>(type); };
    bool anyTensors = false;
    for (auto &op : moduleOp.getOps()) {
      if (op.hasTrait<OpTrait::IREE::Util::ObjectLike>()) {
        continue; // ignore executables
      }
      op.walk([&](Operation *childOp) {
        if (llvm::any_of(childOp->getOperandTypes(), isTensor) ||
            llvm::any_of(childOp->getResultTypes(), isTensor)) {
          anyTensors = true;
          return WalkResult::interrupt();
        }
        return WalkResult::advance();
      });
    }

    //bool anyconvolution = false;
    //  EXSLERATEV2::globalParamas.reset();
    // Value bnMean, bnVar;
    // DenseResourceElementsAttr bnGamma, bnBeta;

    // for(auto &op : moduleOp.getOps()){
    //   op.walk([&](linalg::GenericOp genericOp){

    //     // Find BN variance: Look for AddF and check direct BlockArgument
    //     inputs if(!bnVar){
    //       genericOp.getRegion().walk([&](arith::AddFOp addOp){
    //         auto inputOperands = genericOp.getInputs();
    //         for(auto input : inputOperands){
    //           if(auto funcArg = dyn_cast<BlockArgument>(input)){
    //             if(auto tensorType =
    //             dyn_cast<RankedTensorType>(funcArg.getType())){
    //               if(tensorType.getRank() == 1 &&
    //               mlir::isa<FloatType>(tensorType.getElementType())){
    //                 bnVar = funcArg;
    //                 llvm::outs() << "found bn var " <<
    //                 tensorType.getNumElements() << " elements\n"; return
    //                 WalkResult::interrupt();
    //               }
    //             }
    //           }
    //         }
    //         return WalkResult::advance();
    //       });
    //     }

    //     // Find BN mean: Look for SubF and check expand_shape inputs (same as
    //     gamma/beta pattern) if(!bnMean){
    //       genericOp.getRegion().walk([&](arith::SubFOp subOp){
    //         auto inputOperands = genericOp.getInputs();
    //         for(auto input : inputOperands){
    //           if(auto definingOp = input.getDefiningOp()){
    //             if(auto expandOp =
    //             dyn_cast<tensor::ExpandShapeOp>(definingOp)){
    //               auto expandInput = expandOp.getSrc();
    //               if(auto funcArg = dyn_cast<BlockArgument>(expandInput)){
    //                 if(auto tensorType =
    //                 dyn_cast<RankedTensorType>(funcArg.getType())){
    //                   if(tensorType.getRank() == 1 &&
    //                   mlir::isa<FloatType>(tensorType.getElementType())){
    //                     bnMean = funcArg;
    //                     llvm::errs() << "found bn mean " <<
    //                     tensorType.getNumElements() << " elements\n"; return
    //                     WalkResult::interrupt();
    //                   }
    //                 }
    //               }
    //             }
    //           }
    //         }
    //         return WalkResult::advance();
    //       });
    //     }

    //     // Find BN gamma: Look for MulF and check expand_shape inputs (your
    //     existing working code) if(!bnGamma){
    //       genericOp.getRegion().walk([&](arith::MulFOp mulOp){
    //         auto inputOperands = genericOp.getInputs();
    //         for(auto input : inputOperands){
    //           if(auto definingOp = input.getDefiningOp()){
    //             if(auto expandOp =
    //             dyn_cast<tensor::ExpandShapeOp>(definingOp)){
    //               auto expandInput = expandOp.getSrc();
    //               if(auto constOp =
    //               dyn_cast<arith::ConstantOp>(expandInput.getDefiningOp())){
    //                 if(auto denseAttr =
    //                 dyn_cast<DenseResourceElementsAttr>(constOp.getValue())){
    //                   bnGamma = denseAttr;
    //                   llvm::errs() << "found bn gamma " <<
    //                   denseAttr.getType().getNumElements() << " elements\n";
    //                   return WalkResult::interrupt();
    //                 }
    //               }
    //             }
    //           }
    //         }
    //         return WalkResult::advance();
    //       });
    //     }

    //     // Find BN beta: Look for AddF and check expand_shape inputs (your
    //     existing working code) if(!bnBeta){
    //       genericOp.getRegion().walk([&](arith::AddFOp addOp){
    //         auto inputOperands = genericOp.getInputs();
    //         for(auto input : inputOperands){
    //           if(auto definingOp = input.getDefiningOp()){
    //             if(auto expandOp =
    //             dyn_cast<tensor::ExpandShapeOp>(definingOp)){
    //               auto expandInput = expandOp.getSrc();
    //               if(auto constOp =
    //               dyn_cast<arith::ConstantOp>(expandInput.getDefiningOp())){
    //                 if(auto denseAttr =
    //                 dyn_cast<DenseResourceElementsAttr>(constOp.getValue())){
    //                   if(denseAttr != bnGamma){
    //                     bnBeta = denseAttr;
    //                     llvm::errs() << "found bn beta " <<
    //                     denseAttr.getType().getNumElements() << "
    //                     elements\n"; return WalkResult::interrupt();
    //                   }
    //                 }
    //               }
    //             }
    //           }
    //         }
    //         return WalkResult::advance();
    //       });
    //     }

    //     return WalkResult::advance();
    //   });
    // }

    // bool anyconvolution = false;
    // int layerCount = 0; 

    // for (auto &op : moduleOp.getOps()) {
    //   if (op.hasTrait<OpTrait::IREE::Util::ObjectLike>()) {
    //     continue; 
    //   }

    //   op.walk([&](Operation *childop) {
    //     if (isa<linalg::Conv1DNcwFcwOp, linalg::Conv2DNchwFchwOp,
    //             linalg::Conv2DNhwcFhwcOp, linalg::Conv3DNcdhwFcdhwOp>(
    //             childop)) {
    //       anyconvolution = true;
    //       EXSLERATEV2::LayerParams layerParam;
    //       bool hasweight = false, hasbias = false;

    //       // Extract weights 
    //       if (childop->getNumOperands() >= 2) {
    //         auto weightOperand = childop->getOperand(1);
    //         if (auto definingOp = weightOperand.getDefiningOp()) {
    //           if (auto constOp = dyn_cast<arith::ConstantOp>(definingOp)) {
    //             if (auto denseattr = dyn_cast<DenseResourceElementsAttr>(
    //                     constOp.getValue())) {
    //               layerParam.convWeight = denseattr;
    //               hasweight = true;
    //               llvm::errs()
    //                   << "Extracted convolution weight - Size: "
    //                   << denseattr.getType().getNumElements() << " elements\n";
    //             }
    //           }
    //         }
    //       }

    //       // FIXED: Enhanced bias extraction - multiple patterns
    //       if (childop->getNumOperands() >= 3) {
    //         bool biasFound = false;

    //         // Pattern 1: Your existing logic - last operand with broadcast
    //         auto lastOperand =
    //             childop->getOperand(childop->getNumOperands() - 1);
    //         if (auto broadcastOp = dyn_cast<linalg::BroadcastOp>(
    //                 lastOperand.getDefiningOp())) {
    //           auto biasOperand = broadcastOp.getInput();
    //           if (auto definingOp = biasOperand.getDefiningOp()) {
    //             if (auto constop = dyn_cast<arith::ConstantOp>(definingOp)) {
    //               if (auto denseAttr = dyn_cast<DenseResourceElementsAttr>(
    //                       constop.getValue())) {
    //                 layerParam.convBias = denseAttr;
    //                 hasbias = true;
    //                 biasFound = true;
    //                 llvm::errs()
    //                     << "Extracted convolution bias (broadcast) - Size: "
    //                     << denseAttr.getType().getNumElements()
    //                     << " elements\n";
    //               }
    //             }
    //           }
    //         }

    //         // Pattern 2: Direct constant in last operand (fallback)
    //         if (!biasFound) {
    //           if (auto definingOp = lastOperand.getDefiningOp()) {
    //             if (auto constop = dyn_cast<arith::ConstantOp>(definingOp)) {
    //               if (auto denseAttr = dyn_cast<DenseResourceElementsAttr>(
    //                       constop.getValue())) {
    //                 layerParam.convBias = denseAttr;
    //                 hasbias = true;
    //                 biasFound = true;
    //                 llvm::errs()
    //                     << "Extracted convolution bias (direct) - Size: "
    //                     << denseAttr.getType().getNumElements()
    //                     << " elements\n";
    //               }
    //             }
    //           }
    //         }

    //         // Pattern 3: Check third operand as potential bias
    //         if (!biasFound && childop->getNumOperands() >= 3) {
    //           auto thirdOperand = childop->getOperand(2);
    //           if (auto definingOp = thirdOperand.getDefiningOp()) {
    //             if (auto constop = dyn_cast<arith::ConstantOp>(definingOp)) {
    //               if (auto denseAttr = dyn_cast<DenseResourceElementsAttr>(
    //                       constop.getValue())) {
    //                 // Simple validation: bias should be smaller than weights
    //                 if (hasweight &&
    //                     denseAttr.getType().getNumElements() <
    //                         layerParam.convWeight.getType().getNumElements()) {
    //                   layerParam.convBias = denseAttr;
    //                   hasbias = true;
    //                   llvm::errs()
    //                       << "Extracted convolution bias (3rd operand) - Size: "
    //                       << denseAttr.getType().getNumElements()
    //                       << " elements\n";
    //                 }
    //               }
    //             }
    //           }
    //         }
    //       }

    //       // Store layer if we have weights
    //       if (hasweight) {
    //         EXSLERATEV2::globalParamas.layers.push_back(layerParam);
    //         if (!hasbias) {
    //           llvm::errs() << "Note: Convolution layer " << layerCount
    //                        << " has no bias\n";
    //         }
    //         layerCount++;
    //       } else {
    //         llvm::errs()
    //             << "Warning: Convolution found but no weights extracted\n";
    //       }

    //       return WalkResult::advance();
    //     }
    //     return WalkResult::advance();
    //   });
    // }

    // Enhanced reporting
    // if (anyconvolution) {
    //   llvm::outs() << "Module contains "
    //                << EXSLERATEV2::globalParamas.layers.size()
    //                << " convolution operations with extracted weights\n";

    //   if (EXSLERATEV2::globalParamas.layers.empty()) {
    //     llvm::outs()
    //         << "Warning: Found convolutions but no weights extracted\n";
    //   } else {
    //     for (size_t i = 0; i < EXSLERATEV2::globalParamas.layers.size(); ++i) {
    //       const auto &layer = EXSLERATEV2::globalParamas.layers[i];

    //       if (layer.convWeight) {
    //         llvm::outs() << "Layer " << i << " weight elements: "
    //                      << layer.convWeight.getType().getNumElements() << "\n";
    //       } else {
    //         llvm::outs() << "Layer " << i << " has empty weight\n";
    //       }

    //       if (layer.convBias) {
    //         llvm::outs() << "Layer " << i << " bias elements: "
    //                      << layer.convBias.getType().getNumElements() << "\n";
    //       } else {
    //         llvm::outs() << "Layer " << i << " has no bias\n";
    //       }

    //       // Your existing BN parameter reporting...
    //       if (layer.bnGamma) {
    //         llvm::outs() << "Layer " << i << " BN gamma elements: "
    //                      << layer.bnGamma.getType().getNumElements() << "\n";
    //       } else {
    //         llvm::outs() << "Layer " << i << " has no BN gamma\n";
    //       }

    //       if (layer.bnBeta) {
    //         llvm::outs() << "Layer " << i << " BN beta elements: "
    //                      << layer.bnBeta.getType().getNumElements() << "\n";
    //       } else {
    //         llvm::outs() << "Layer " << i << " has no BN beta\n";
    //       }

    //       if (layer.bnMean) {
    //         auto meanType = dyn_cast<RankedTensorType>(layer.bnMean.getType());
    //         if (meanType) {
    //           llvm::outs() << "Layer " << i
    //                        << " BN mean elements: " << meanType.getNumElements()
    //                        << "\n";
    //         }
    //       } else {
    //         llvm::outs() << "Layer " << i << " has no BN mean\n";
    //       }

    //       if (layer.bnVar) {
    //         auto varType = dyn_cast<RankedTensorType>(layer.bnVar.getType());
    //         if (varType) {
    //           llvm::outs() << "Layer " << i
    //                        << " BN var elements: " << varType.getNumElements()
    //                        << "\n";
    //         }
    //       } else {
    //         llvm::outs() << "Layer " << i << " has no BN var\n";
    //       }
    //     }
    //   }
    // } else {
    //   llvm::outs() << "No convolution operations found in module\n";
    // }

    // TODO(multi-device): the logic above is insufficient; we only need devices
    // if the program will end up requiring them but we don't know that here.
    // We have to wait until we've lowered to the point where we do require a
    // device _and_ we actually want one (aren't compiling a non-HAL program).
    // We could probably have an op interface, better output from the pass that
    // requires the devices, etc. For now we error out in HAL conversion when we
    // try to resolve devices.
    if (false && anyTensors && deviceAnalysis.getDeviceGlobals().empty()) {
      auto diagnostic = moduleOp.emitError();
      diagnostic
          << "no HAL devices defined in the module; use the module-level "
             "hal.device.targets attribute, the --iree-hal-target-device= "
             "flag, or provide inputs with global !hal.devices defined; ";
      printAvailable(diagnostic, *targetRegistry.value);
      return signalPassFailure();
    }

    // Walk all devices and verify them.
    for (auto deviceOp : deviceAnalysis.getDeviceGlobals()) {
      if (auto initialValue = deviceOp.getGlobalInitialValue()) {
        if (failed(verifyAttr(deviceOp, initialValue, *targetRegistry.value))) {
          return signalPassFailure();
        }
      }
    }
  }
};

} // namespace

} // namespace mlir::iree_compiler::IREE::HAL

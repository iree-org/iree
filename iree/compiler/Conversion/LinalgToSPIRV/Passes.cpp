// Copyright 2020 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//===- Passes.cpp - Pipeline from HLO to Linalg to SPIR-V -----------------===//
//
// Implementation of conversion from XLA-HLO to Linalg to SPIR-V dialect.
//
//===----------------------------------------------------------------------===//

#include "iree/compiler/Conversion/LinalgToSPIRV/Passes.h"

#include "iree/compiler/Conversion/CodegenUtils/ForOpCanonicalization.h"
#include "iree/compiler/Conversion/Common/Passes.h"
#include "iree/compiler/Conversion/HLOToHLO/Passes.h"
#include "iree/compiler/Conversion/HLOToLinalg/Passes.h"
#include "iree/compiler/Conversion/LinalgToSPIRV/CodeGenOptionUtils.h"
#include "iree/compiler/Conversion/LinalgToSPIRV/MemorySpace.h"
#include "iree/compiler/Conversion/LinalgToVector/Passes.h"
#include "iree/compiler/Dialect/Shape/Transforms/Passes.h"
#include "llvm/Support/CommandLine.h"
#include "mlir/Conversion/GPUToSPIRV/GPUToSPIRV.h"
#include "mlir/Conversion/SCFToGPU/SCFToGPUPass.h"
#include "mlir/Conversion/StandardToSPIRV/StandardToSPIRV.h"
#include "mlir/Conversion/StandardToSPIRV/StandardToSPIRVPass.h"
#include "mlir/Dialect/GPU/GPUDialect.h"
#include "mlir/Dialect/GPU/Passes.h"
#include "mlir/Dialect/Linalg/IR/LinalgOps.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVOps.h"
#include "mlir/Dialect/SPIRV/Transforms/Passes.h"
#include "mlir/Dialect/SPIRV/Transforms/SPIRVConversion.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassOptions.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Transforms/FoldUtils.h"
#include "mlir/Transforms/Passes.h"

namespace mlir {
namespace iree_compiler {

static llvm::cl::opt<bool> clEnableLinalgOnTensorsSPIRV(
    "iree-codegen-spirv-experimental-linalg-on-tensors",
    llvm::cl::desc("Enable the linalg on tensors on SPIR-V path"),
    llvm::cl::init(false));

static void addLinalgToSPIRVPasses(OpPassManager &pm,
                                   const SPIRVCodegenOptions &options) {
  //===--------------------------------------------------------------------===//
  // Initial clean up.
  //===--------------------------------------------------------------------===//
  pm.nest<ModuleOp>().addPass(createCanonicalizerPass());
  pm.nest<ModuleOp>().addPass(createCSEPass());

  //===--------------------------------------------------------------------===//
  // Tile Linalg on buffers.
  //
  // Pre-conditions:
  //   - All Linalg ops have buffer semantics.
  //
  // Post-conditions:
  //   - If there are multiple linalg operations in the dispatch region, they
  //     are fused, using tile+fuse approach.
  //     - The fused loops are distributed across workgroups.
  //   - The operations that cannot be fused at buffer levels are split into
  //     separate entry points.
  //   - If there is a single linalg operation in the dispatch region, it is
  //     tiled and the generated parallel loop distributed.
  //     - The tiled linalg operation can be tiled again one or more times and
  //       then vectorized.
  //   - Otherwise:
  //     - The Linalg op is kept untouched.
  //
  //===--------------------------------------------------------------------===//
  if (!clEnableLinalgOnTensorsSPIRV) {
    pm.nest<ModuleOp>().addPass(createSplitDispatchFunctionPass());
  }
  pm.addPass(createLinalgTileAndFusePass(options));
  if (options.vectorizeMemref) {
    pm.nest<ModuleOp>().addNestedPass<FuncOp>(
        createLoadStoreVectorizationPass());
  }
  pm.nest<ModuleOp>().addPass(createCanonicalizerPass());

  //===--------------------------------------------------------------------===//
  // Map to GPU processor IDs.
  //
  // Post-conditions:
  //   - loop.parallel ops are converted to loop.for ops and mapped to
  //     workgroups.
  //   - Linalg ops are converted to loop.for ops and mapped to workitems.
  //===--------------------------------------------------------------------===//
  pm.nest<ModuleOp>().addPass(createConvertToGPUPass());
  if (options.enableVectorization) {
    pm.nest<ModuleOp>().addNestedPass<FuncOp>(createVectorToGPUPass());
  }
  pm.nest<ModuleOp>().addPass(createLowerAffinePass());
  pm.nest<ModuleOp>().addPass(createCanonicalizerPass());
  pm.nest<ModuleOp>().addPass(createCSEPass());

  if (!clEnableLinalgOnTensorsSPIRV) {
    //===--------------------------------------------------------------------===//
    // Legalize the function that computes the number of workgroups to be
    // runnable on the host.
    //
    // Post-conditions:
    //   - The shape of the values created from `iree.placeholder` operations
    //   are
    //     tied to the arguments of the function.
    //===--------------------------------------------------------------------===//
    pm.nest<ModuleOp>().addPass(createLegalizeNumWorkgroupsFnPass());

    //===--------------------------------------------------------------------===//
    // Resolve shape related ops.
    //
    // Pre-conditions:
    //   - All dynamic tensors bridge through a shapex.tie_shape op with the
    //     appropriate shape.
    //   - No shapex.get_ranked_shape ops exist.
    //   - Shape folding and canonicalization has been done.
    // Post-conditions:
    //   - shapex.tie_shape and other shapex ops are all converted away.
    //   - std.dim ops are traced back and replaced by the corresponding
    //     hal.inteface.load.constant op. There are no std.dim ops left
    //     in the IR.
    //===--------------------------------------------------------------------===//
    pm.nest<ModuleOp>().addNestedPass<FuncOp>(createResolveShapeOpsPass());

    //===--------------------------------------------------------------------===//
    // Legalize the function that computes the number of workgroups to be
    // runnable on the host.
    //
    // Post-conditions:
    //   - The dead `iree.placeholder` operations are removed after shape
    //     resolution.
    //===--------------------------------------------------------------------===//
    pm.nest<ModuleOp>().addPass(createLegalizeNumWorkgroupsFnPass());
  }

  //===--------------------------------------------------------------------===//
  // Prepare stdandard ops for SPIR-V conversion.
  //
  // Post-conditions:
  //   - Load/store on std.subview ops are converted into load/store on the
  //     original buffers.
  //===--------------------------------------------------------------------===//
  if (options.enableVectorization) {
    pm.nest<ModuleOp>().addNestedPass<FuncOp>(
        createVectorTransferOptimizationPass());
  }
  pm.nest<ModuleOp>().addPass(createLegalizeStdOpsForSPIRVLoweringPass());
  pm.nest<ModuleOp>().addPass(createCanonicalizerPass());
  pm.nest<ModuleOp>().addPass(createCSEPass());
  if (options.enableVectorization) {
    pm.nest<ModuleOp>().addPass(createVectorizeMemref());
    pm.nest<ModuleOp>().addNestedPass<FuncOp>(
        createForOpCanonicalizationPass());
    pm.nest<ModuleOp>().addPass(createCanonicalizerPass());
    pm.nest<ModuleOp>().addPass(createCSEPass());
  }

  //===--------------------------------------------------------------------===//
  // Final conversion to SPIR-V dialect.
  //
  // Post-conditions:
  //   - All ops are converted to SPIR-V counterparts.
  //   - spv.module ops are formed to hold all SPIR-V ops.
  //===--------------------------------------------------------------------===//
  pm.nest<ModuleOp>().addPass(createConvertToSPIRVPass());

  //===--------------------------------------------------------------------===//
  // SPIR-V dialect level conversions.
  //
  // Post-conditions:
  //   - SPIR-V Entry point ops are inserted.
  //   - Required version/extension/capability are deduced.
  //===--------------------------------------------------------------------===//
  OpPassManager &spirvModulePM = pm.nest<ModuleOp>().nest<spirv::ModuleOp>();
  spirvModulePM.addPass(spirv::createLowerABIAttributesPass());
  spirvModulePM.addPass(createCanonicalizerPass());
  spirvModulePM.addPass(createCSEPass());
  spirvModulePM.addPass(spirv::createUpdateVersionCapabilityExtensionPass());
}

void buildSPIRVTransformPassPipeline(OpPassManager &pm,
                                     const SPIRVCodegenOptions &options) {
  //===--------------------------------------------------------------------===//
  // The entry point functions call an _impl function that captures the ABI that
  // the host side uses for the dispatch region. This ABI is needed when
  // generating the function that computes the number of workgroups. Declare the
  // function that returns the number of workgroups needed for an entry point
  // function.
  //
  // Post-conditions

  //   - An empty, private function is defined for each entry point function
  //     that returns the number of workgroups.
  //   - The entry point function gets an attribute `vkspv.num_workgroups_fn` to
  //     record which function in the module returns the number of workgroups.
  if (!clEnableLinalgOnTensorsSPIRV) {
    pm.nest<ModuleOp>().addPass(createDeclareNumWorkgroupsFnPass());
  }

  //===--------------------------------------------------------------------===//
  // Inline the impl dispatch function into the wrapper dispatch function.
  //
  // TODO(antiagainst): re-evaluate the inlining timing.
  //===--------------------------------------------------------------------===//
  pm.nest<ModuleOp>().addPass(createInlinerPass());

  if (clEnableLinalgOnTensorsSPIRV) {
    WorkgroupMemoryAllocationFn allocationFn = [](OpBuilder &builder,
                                                  Location loc,
                                                  ArrayRef<Value> dynamicSizes,
                                                  MemRefType allocationType) {
      MemRefType allocType = MemRefType::get(allocationType.getShape(),
                                             allocationType.getElementType(),
                                             {}, getWorkgroupMemorySpace());
      return builder.create<AllocOp>(loc, allocType, dynamicSizes);
    };
    addLinalgBufferizePasses(pm.nest<ModuleOp>(), allocationFn);
  } else {
    //===--------------------------------------------------------------------===//
    // Inject shape calculation for output buffers.
    //
    // Pre-conditions:
    //   - All transformations altering the tensor-level shapes have been done.
    //   - "Root" dynamic tensors all pass through a single shapex.tie_shape
    //     use which associates them to their shape.
    //   - Loose, non-associated shapex.get_ranked_shape ops can exist anywhere
    //     and will be resolved.
    // Post-conditions:
    //   - All dynamic tensors bridge through a shapex.tie_shape op with the
    //     appropriate shape.
    //   - No shapex.get_ranked_shape ops exist.
    //   - Shape folding and canonicalization has been done.
    //===--------------------------------------------------------------------===//
    pm.nest<ModuleOp>().addNestedPass<FuncOp>(
        Shape::createTieDynamicShapesPass());
    pm.nest<ModuleOp>().addNestedPass<FuncOp>(
        Shape::createMaterializeShapeCalculationsPass());
    pm.nest<ModuleOp>().addNestedPass<FuncOp>(
        Shape::createHoistShapeCalculationsPass());

    //===--------------------------------------------------------------------===//
    // Convert XLA HLO ops to Linalg ops with buffer semantics.
    //
    // Post-conditions:
    //   - All XLA HLO ops are converted.
    //   - All Linalg ops are operating on buffers.
    //===--------------------------------------------------------------------===//
    pm.nest<ModuleOp>().addNestedPass<FuncOp>(createDecomposeHLOClampPass());
    addHLOToLinalgOnBuffersPasses(pm.nest<ModuleOp>());
  }

  //===--------------------------------------------------------------------===//
  // Convert Linalg ops to SPIR-V ops.
  //
  // Post-conditions:
  //   - All Linalg/Loops/GPU/Affine/Standard ops are converted away.
  //   - The module contains the final spv.module ready for serialization.
  //===--------------------------------------------------------------------===//
  addLinalgToSPIRVPasses(pm, options);

  if (!clEnableLinalgOnTensorsSPIRV) {
    // HACK: SplitDispatchFunctionPass inserts spv.EntryPoints but does not tell
    // the HAL about them. We need to find those new entry points and
    // materialize hal.executable.entry_point ops so that we have a consistent
    // view of the executable.  SplitDispatchFunctionPass can hopefully go away
    // with linalg-on-tensors and we can remove this.
    pm.addPass(createMaterializeEntryPointsPass());
  }
}

static PassPipelineRegistration<> linalgToSPIRVPipeline(
    "iree-codegen-linalg-to-spirv-pipeline",
    "Runs the progressive lowering pipeline from Linalg to SPIR-V",
    [](OpPassManager &passManager) {
      addLinalgToSPIRVPasses(passManager,
                             getSPIRVCodegenOptionsFromClOptions());
    });

static PassPipelineRegistration<> hloToLinalgSPIRVPipeline(
    "iree-codegen-hlo-to-spirv-pipeline",
    "Runs the progressive lowering pipeline from XLA HLO to Linalg to "
    "SPIR-V",
    [](OpPassManager &passManager) {
      buildSPIRVTransformPassPipeline(passManager,
                                      getSPIRVCodegenOptionsFromClOptions());
    });

}  // namespace iree_compiler
}  // namespace mlir

// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree-dialects/Dialect/LinalgExt/IR/LinalgExtDialect.h"
#include "iree-dialects/Dialect/LinalgExt/TransformOps/LinalgExtTransformOps.h"
#include "iree-dialects/Dialect/LinalgTransform/LinalgTransformOps.h"
#include "iree-dialects/Dialect/LinalgTransform/StructuredTransformOpsExt.h"
#include "iree-dialects/Dialect/LinalgTransform/TransformInterpreterUtils.h"
#include "iree/compiler/Codegen/Common/TransformExtensions/CommonExtensions.h"
#include "iree/compiler/Codegen/LLVMCPU/TransformExtensions/LLVMCPUExtensions.h"
#include "iree/compiler/Codegen/LLVMGPU/TransformExtensions/LLVMGPUExtensions.h"
#include "iree/compiler/Codegen/PassDetail.h"
#include "iree/compiler/Codegen/Passes.h"
#include "iree/compiler/Dialect/Flow/IR/FlowDialect.h"
#include "iree/compiler/Dialect/Flow/TransformExtensions/FlowExtensions.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/TransformOps/AffineTransformOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Bufferization/Transforms/FuncBufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/GPU/TransformOps/GPUTransformOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/TransformOps/LinalgTransformOps.h"
#include "mlir/Dialect/Linalg/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/MemRef/TransformOps/MemRefTransformOps.h"
#include "mlir/Dialect/PDL/IR/PDL.h"
#include "mlir/Dialect/PDLInterp/IR/PDLInterp.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/TransformOps/SCFTransformOps.h"
#include "mlir/Dialect/SCF/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tensor/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Transform/IR/TransformDialect.h"
#include "mlir/Dialect/Transform/IR/TransformInterfaces.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/Vector/TransformOps/VectorTransformOps.h"
#include "mlir/Dialect/Vector/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Support/FileUtilities.h"

#define DEBUG_TYPE "iree-transform-dialect-interpreter"
#define DEBUG_TYPE_DUMP_STDERR "iree-transform-dialect-dump-repro"
#define DEBUG_TYPE_DUMP_FILE "iree-transform-dialect-save-repro"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")

using namespace mlir;

/// Finds the single top-level transform operation with `root` as ancestor.
/// Reports an error if there is more than one such operation and returns the
/// first one found. Reports an error returns nullptr if no such operation
/// found.
static Operation *findTopLevelTransform(Operation *root, StringRef debugStr) {
  transform::TransformOpInterface topLevelTransform = nullptr;
  WalkResult walkResult = root->walk<WalkOrder::PreOrder>(
      [&](transform::TransformOpInterface transformOp) {
        if (!topLevelTransform) {
          topLevelTransform = transformOp;
          return WalkResult::skip();
        }
        auto diag = transformOp.emitError()
                    << "more than one top-level transform op";
        diag.attachNote(topLevelTransform.getLoc())
            << "previous top-level transform op";
        return WalkResult::interrupt();
      });
  if (walkResult.wasInterrupted()) return nullptr;
  if (!topLevelTransform) {
    auto diag = root->emitError()
                << "could not find a nested top-level transform op";
    diag.attachNote() << "use the '" << debugStr
                      << "' option to provide transform as external file";
    return nullptr;
  }
  return topLevelTransform;
}

/// Finds an operation nested in `root` that has the transform dialect tag
/// attribute with the value specified as `tag`. Assumes only one operation
/// may have the tag. Returns nullptr if there is no such operation.
static Operation *findOpWithTag(Operation *root, StringRef tagKey,
                                StringRef tagValue) {
  Operation *found = nullptr;
  root->walk<WalkOrder::PreOrder>([tagKey, tagValue, &found](Operation *op) {
    auto attr = op->getAttrOfType<StringAttr>(tagKey);
    if (!attr || attr.getValue() != tagValue) return WalkResult::advance();

    assert(found == nullptr && "more than one op with the same tag");
    found = op;

    // In debug mode, continue the traversal to see if the tag is not
    // duplicated.
#ifndef NDEBUG
    return WalkResult::advance();
#else
    return WalkResult::interrupt();
#endif  // NDEBUG
  });
  return found;
}

namespace {

/// Pass declaration.
/// Interpreter pass that applies transform dialect ops for codegen.
/// This needs to be its own pass because the registration mechanism and ops
/// available are different than for other interpreters.
class TransformDialectInterpreterPass
    : public iree_compiler::TransformDialectInterpreterBase<
          TransformDialectInterpreterPass> {
 public:
  void getDependentDialects(DialectRegistry &registry) const override {
    // TODO: this is only necessary to make registry subset happy when running
    // the lowering to LLVM. The lowering should be changed to stop using the
    // nested pass manager and this will go away.

    // clang-format off
    registry.insert<mlir::iree_compiler::IREE::LinalgExt::IREELinalgExtDialect,
                    mlir::iree_compiler::IREE::Flow::FlowDialect,
                    arith::ArithDialect,
                    AffineDialect,
                    bufferization::BufferizationDialect,
                    func::FuncDialect,
                    gpu::GPUDialect,
                    linalg::LinalgDialect,
                    linalg::transform::LinalgTransformDialect,
                    LLVM::LLVMDialect,
                    pdl::PDLDialect,
                    pdl_interp::PDLInterpDialect,
                    scf::SCFDialect,
                    tensor::TensorDialect,
                    transform::TransformDialect,
                    vector::VectorDialect
        // clang-format on
        >();

    // TODO: these should be registered by the extension instead, but there is
    // no support for it in core currently.
    arith::registerBufferizableOpInterfaceExternalModels(registry);
    linalg::registerBufferizableOpInterfaceExternalModels(registry);
    scf::registerBufferizableOpInterfaceExternalModels(registry);
    bufferization::func_ext::registerBufferizableOpInterfaceExternalModels(
        registry);
    tensor::registerBufferizableOpInterfaceExternalModels(registry);
    vector::registerBufferizableOpInterfaceExternalModels(registry);

    registry.addExtensions<
        mlir::iree_compiler::IREE::LinalgExt::LinalgExtTransformOpsExtension,
        transform_ext::StructuredTransformOpsExtension>();
    iree_compiler::registerTransformDialectCommonExtension(registry);
    iree_compiler::registerTransformDialectFlowExtension(registry);
    iree_compiler::registerTransformDialectLLVMCPUExtension(registry);
    iree_compiler::registerTransformDialectLLVMGPUExtension(registry);
    affine::registerTransformDialectExtension(registry);
    gpu::registerTransformDialectExtension(registry);
    linalg::registerTransformDialectExtension(registry);
    memref::registerTransformDialectExtension(registry);
    scf::registerTransformDialectExtension(registry);
    vector::registerTransformDialectExtension(registry);
  }

  TransformDialectInterpreterPass(
      StringRef transformFileName = StringRef(),
      StringRef debugPayloadRootTag = StringRef(),
      StringRef debugTransformRootTag = StringRef()) {
    this->transformFileName = transformFileName.str();
    this->debugPayloadRootTag = debugPayloadRootTag.str();
    this->debugTransformRootTag = debugTransformRootTag.str();
  }
  TransformDialectInterpreterPass(const TransformDialectInterpreterPass &pass) {
    this->transformFileName = pass.transformFileName;
    this->debugPayloadRootTag = pass.debugPayloadRootTag;
    this->debugTransformRootTag = pass.debugTransformRootTag;
    // TODO: if we really don't like shared_ptr, we could also clone the
    // transformModule here.
    sharedTransformModule = pass.sharedTransformModule;
  }

  LogicalResult initialize(MLIRContext *context) override {
    OwningOpRef<ModuleOp> module;
    if (failed(transform::parseTransformModuleFromFile(
            context, transformFileName, module)))
      return failure();

    sharedTransformModule =
        std::make_shared<OwningOpRef<ModuleOp>>(std::move(module));
    return success();
  }

  void runOnOperation() override;

 private:
  // Optionally perform debug actions requested by the user to dump IR and a
  // repro to stderr and/or a fie.
  void performOptionalDebugActions(Operation *target, Region *transformRegion);

  /// Name of the attribute used for targeting the transform dialect interpreter
  /// at specific operations.
  constexpr static llvm::StringLiteral kTransformIreeTagAttrName =
      "transform.iree_tag";
  /// Value of the attribute indicating the root payload operation.
  constexpr static llvm::StringLiteral kTransformIreeTagPayloadRootValue =
      "iree_payload_root";
  /// Value of the attribute indicating the container of transform operations
  /// (containing the top-level transform operation).
  constexpr static llvm::StringLiteral
      kTransformIreeTagTransformContainerValue = "iree_transform_container";

  /// Returns the ancestor of `target` that doesn't have a parent.
  Operation *getRootOperation(Operation *target) {
    Operation *root = target;
    while (root->getParentOp()) root = root->getParentOp();
    return root;
  }

  /// Prints the CLI command running the repro with the current path.
  llvm::raw_ostream &printIreeOptReproCall(llvm::raw_ostream &os,
                                           StringRef rootOpName);

  /// Prints the module rooted at `root` to `os` and appends
  /// `transformContainer` if it is not nested in `root`.
  llvm::raw_ostream &printModuleForRepro(llvm::raw_ostream &os, Operation *root,
                                         Operation *transformContainer);

  /// Saves the payload and the transform IR into a temporary file and reports
  /// the file name to `os`.
  void saveReproToTempFile(llvm::raw_ostream &os, Operation *target,
                           Operation *transformContainer);

  // The parsed transform module to be used for transformations.
  // TODO: Figure a better way to build a transform module and transport it in
  // the proper places in the IR as it is transformed by IREE so that it is
  // available with better ownership semantics.
  // Note: we wrap the OwningOpRef to get the desired destruction mechanism.
  // Note: shared_ptr is not great but we know the sharedTransformModule is
  // readonly.
  // Alternatives comprise:
  //   1. no shared_ptr but copying the module with every pass clone that the
  //      OpPassManager decides to perform.
  //   2. lifting ownership of the parsed transform module higher up in the
  //      IREE stack. This may be only shift the problem as we have passes
  //      building pass managers in IREE.
  //   3. build better support to embed the transformation module in the
  //      input IR and transport it to the place of use in IREE. This is deemed
  //      too intrusive atm.
  //   4. (future) config/resources mechanism that is being proposed in core?
  std::shared_ptr<OwningOpRef<ModuleOp>> sharedTransformModule;
};
}  // namespace

/// Prints the CLI command running the repro with the current path.
llvm::raw_ostream &TransformDialectInterpreterPass::printIreeOptReproCall(
    llvm::raw_ostream &os, StringRef rootOpName) {
  os << llvm::formatv(
      "iree-opt "
      "--pass-pipeline=\"{0}(iree-transform-dialect-interpreter{{{1}={2} "
      "{3}={4}})\"",
      rootOpName, debugPayloadRootTag.getArgStr(),
      debugPayloadRootTag.empty() ? StringRef(kTransformIreeTagPayloadRootValue)
                                  : debugPayloadRootTag,
      debugTransformRootTag.getArgStr(),
      debugTransformRootTag.empty()
          ? StringRef(kTransformIreeTagTransformContainerValue)
          : debugTransformRootTag);
  return os;
}

/// Prints the module rooted at `root` to `os` and appends
/// `transformContainer` if it is not nested in `root`.
llvm::raw_ostream &TransformDialectInterpreterPass::printModuleForRepro(
    llvm::raw_ostream &os, Operation *root, Operation *transformContainer) {
  root->print(os);
  if (!root->isAncestor(transformContainer)) {
    transformContainer->print(os);
  }
  return os;
}

/// Saves the payload and the transform IR into a temporary file and reports
/// the file name to `os`.
void TransformDialectInterpreterPass::saveReproToTempFile(
    llvm::raw_ostream &os, Operation *target, Operation *transformContainer) {
  using llvm::sys::fs::TempFile;
  Operation *root = getRootOperation(target);

  SmallVector<char, 128> tmpPath;
  llvm::sys::path::system_temp_directory(/*erasedOnReboot=*/true, tmpPath);
  llvm::sys::path::append(tmpPath, "iree_transform_dialect_%%%%%%.mlir");
  llvm::Expected<TempFile> tempFile = TempFile::create(tmpPath);
  if (!tempFile) {
    os << "could not open temporary file to save the repro\n";
    return;
  }

  llvm::raw_fd_ostream fout(tempFile->FD, /*shouldClose=*/false);
  printModuleForRepro(fout, root, transformContainer);
  fout.flush();
  std::string filename = tempFile->TmpName;

  if (tempFile->keep()) {
    os << "could not preserve the temporary file with the repro\n";
    return;
  }

  os << "=== Transform Interpreter Repro ===\n";
  printIreeOptReproCall(os, root->getName().getStringRef())
      << " " << filename << "\n";
  os << "===================================\n";
}

// Optionally perform debug actions requested by the user to dump IR and a
// repro to stderr and/or a fie.
void TransformDialectInterpreterPass::performOptionalDebugActions(
    Operation *target, Region *transformRegion) {
  // Add temporary debug / repro attributes, these must never leak out.
  if (debugPayloadRootTag.empty()) {
    target->setAttr(
        kTransformIreeTagAttrName,
        StringAttr::get(&getContext(), kTransformIreeTagPayloadRootValue));
  }
  if (debugTransformRootTag.empty()) {
    transformRegion->getParentOp()->setAttr(
        kTransformIreeTagAttrName,
        StringAttr::get(&getContext(),
                        kTransformIreeTagTransformContainerValue));
  }

  DEBUG_WITH_TYPE(DEBUG_TYPE_DUMP_STDERR, {
    Operation *root = getRootOperation(target);
    llvm::dbgs() << "=== Transform Interpreter Repro ===\n";
    printIreeOptReproCall(llvm::dbgs() << "cat <<EOF | ",
                          root->getName().getStringRef());
    printModuleForRepro(llvm::dbgs(), root, transformRegion->getParentOp());
    llvm::dbgs() << "\nEOF\n";
    llvm::dbgs() << "===================================\n";
  });
  DEBUG_WITH_TYPE(DEBUG_TYPE_DUMP_FILE, {
    saveReproToTempFile(llvm::dbgs(), target, transformRegion->getParentOp());
  });

  // Drop the temporary debug / repro attributes, these must never leak out.
  if (debugTransformRootTag.empty()) {
    transformRegion->getParentOp()->removeAttr(
        kTransformIreeTagTransformContainerValue);
  }
  if (debugPayloadRootTag.empty()) {
    target->removeAttr(kTransformIreeTagAttrName);
  }
}

void TransformDialectInterpreterPass::runOnOperation() {
  Operation *target = getOperation();
  bool parsedTransform = (sharedTransformModule && *sharedTransformModule);

  // Step 1
  // ------
  // Get the default payloadRoot and transformRegion that one expects
  // when running the IREE nested pass pipeline or the interpreter.
  Operation *payloadRoot = target;
  Region *transformRegion = nullptr;
  // If a parsed transform was specified separately, use it immediately.
  // Otherwise, the transform is embedded in the IR: go inspect the IR and
  // get the first top-level transform we find.
  if (parsedTransform) {
    transformRegion = &(*sharedTransformModule)->getRegion();
  } else {
    // TODO: In large IR we will likely want more control in selecting a
    // particular transform to focus on, this may warrant a user-specified
    // attribute that one would manually injected in the IR when operating in
    // interpreted mode.
    Operation *topLevelTransform =
        findTopLevelTransform(target, transformFileName.getArgStr());
    if (!topLevelTransform) return signalPassFailure();
    transformRegion = topLevelTransform->getParentRegion();
  }
  assert(transformRegion && "unexpected detached root transform op");

  // Step 2
  // ------
  // Optionally override payloadRoot if the debugPayloadRootTag was passed.
  //
  // If debugPayloadRootTag was passed, then we are in user-specified selection
  // of the transformed IR. This corresponds to REPL debug mode.
  // Otherwise, just apply to `target`, which is what the IREE nested
  // pipeline wants to operate on.
  if (!debugPayloadRootTag.empty()) {
    payloadRoot =
        findOpWithTag(target, kTransformIreeTagAttrName, debugPayloadRootTag);
    if (!payloadRoot) {
      target->emitError() << "couldn't find the root payload op with "
                          << kTransformIreeTagAttrName << "=\""
                          << kTransformIreeTagPayloadRootValue
                          << "\" attribute";
      return signalPassFailure();
    }
  }

  // Step 3
  // ------
  // Optionally override transformRegion if the debugTransformRootTag was
  // passed.
  //
  // If debugTransformRootTag was passed, then we are in user-specified
  // selection of the transforming IR. This corresponds to REPL debug mode.
  // Otherwise, just apply to the existing `transformRegion`, which is what
  // the IREE nested pipeline wants to operate on.
  if (!debugTransformRootTag.empty()) {
    Operation *transformRoot =
        findOpWithTag(transformRegion->getParentOp(), kTransformIreeTagAttrName,
                      kTransformIreeTagTransformContainerValue);
    if (!transformRoot) {
      transformRegion->getParentOp()->emitError()
          << "couldn't find the transform container op with "
          << kTransformIreeTagAttrName << "=\""
          << kTransformIreeTagTransformContainerValue << "\" attribute";
      return signalPassFailure();
    }
    if (transformRoot->getNumRegions() != 1 ||
        !transformRoot->getRegion(0).hasOneBlock()) {
      transformRoot->emitError() << "expected transform container op to have "
                                    "one single-block region";
      return signalPassFailure();
    }
    transformRegion = &transformRoot->getRegion(0);
  }

  // Step 4
  // ------
  // Optionally perform debug actions requested by the user to dump IR and a
  // repro to stderr and/or a fie.
  performOptionalDebugActions(target, transformRegion);

  // Step 5
  // ------
  // Apply the transform to the IR
  // TODO: lift this assertion.
  assert(transformRegion->getBlocks().size() == 1 &&
         "expected single-region block");
  if (failed(
          transform::applyTransformsInRegion(*transformRegion, payloadRoot))) {
    payloadRoot->emitOpError() << "transform dialect interpreter failed";
    return signalPassFailure();
  }
}

namespace mlir {
namespace iree_compiler {
/// Create a Transform dialect interpreter pass.
std::unique_ptr<Pass> createTransformDialectInterpreterPass(
    llvm::StringRef transformFileName, llvm::StringRef debugPayloadRootTag,
    llvm::StringRef debugTransformRootTag) {
  return std::make_unique<TransformDialectInterpreterPass>(
      transformFileName, debugPayloadRootTag, debugTransformRootTag);
}
}  // namespace iree_compiler
}  // namespace mlir

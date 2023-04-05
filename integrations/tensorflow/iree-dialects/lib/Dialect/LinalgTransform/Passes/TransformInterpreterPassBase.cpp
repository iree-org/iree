// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree-dialects/Dialect/LinalgTransform/TransformInterpreterPassBase.h"
#include "mlir/Dialect/Transform/IR/TransformInterfaces.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/FileUtilities.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;

#define DEBUG_TYPE "transform-dialect-interpreter"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE << "]: ")
#define DEBUG_TYPE_DUMP_STDERR "iree-transform-dialect-dump-repro"
#define DEBUG_TYPE_DUMP_FILE "iree-transform-dialect-save-repro"

/// Name of the attribute used for targeting the transform dialect interpreter
/// at specific operations.
constexpr static llvm::StringLiteral kTransformIreeTagAttrName =
    "transform.iree_tag";
/// Value of the attribute indicating the root payload operation.
constexpr static llvm::StringLiteral kTransformIreeTagPayloadRootValue =
    "iree_payload_root";
/// Value of the attribute indicating the container of transform operations
/// (containing the top-level transform operation).
constexpr static llvm::StringLiteral kTransformIreeTagTransformContainerValue =
    "iree_transform_container";

/// Utility to parse the content of a `transformFileName` mlir file containing
/// a transform dialect specification.
static LogicalResult
parseTransformModuleFromFile(MLIRContext *context,
                             llvm::StringRef transformFileName,
                             OwningOpRef<ModuleOp> &transformModule) {
  if (transformFileName.empty()) {
    LLVM_DEBUG(
        DBGS() << "no transform file name specified, assuming the transform "
                  "module is embedded in the IR next to the top-level\n");
    return success();
  }
  // Parse transformFileName content into a ModuleOp.
  std::string errorMessage;
  auto memoryBuffer = mlir::openInputFile(transformFileName, &errorMessage);
  if (!memoryBuffer) {
    llvm::errs() << "failed to parse transform file: " << transformFileName
                 << "\n";
    return failure();
  }
  // Tell sourceMgr about this buffer, the parser will pick it up.
  llvm::SourceMgr sourceMgr;
  sourceMgr.AddNewSourceBuffer(std::move(memoryBuffer), llvm::SMLoc());
  transformModule =
      OwningOpRef<ModuleOp>(parseSourceFile<ModuleOp>(sourceMgr, context));
  return success();
}

/// Utility to extract the `TransformOpInterface` ops that have the trait
/// `PossibleTopLevelTransformOpTrait`.
static LogicalResult extractTopLevelTransformOps(
    Region &r, SmallVectorImpl<transform::TransformOpInterface> &res) {
  assert(r.getBlocks().size() == 1 &&
         "Expected single-block region to extract transform ops from");
  r.walk<WalkOrder::PreOrder>([&](transform::TransformOpInterface transform) {
    if (transform->hasTrait<transform::PossibleTopLevelTransformOpTrait>()) {
      assert(llvm::none_of(res, [&](transform::TransformOpInterface seen) {
        return seen->isAncestor(transform);
      }));
      res.push_back(transform);
      return WalkResult::skip();
    }
    return WalkResult::advance();
  });
  return success();
}

/// Utility to run a transform dialect specification contained in a
/// `transformRegion`, on a `target` op.
/// Since the transform dialect may use PDL which may modify the IR, the
/// underlying implementation clones the transform dialect operations before
/// applying them.
static LogicalResult applyTransformsInRegion(Region &transformRegion,
                                             Operation *target) {
  SmallVector<transform::TransformOpInterface> transforms;
  if (failed(extractTopLevelTransformOps(transformRegion, transforms)))
    return failure();

  for (transform::TransformOpInterface transform : transforms) {
    // TransformState::applyTransform requires that the parent region is a
    // proper ancestor of the transform op to perform SSA liveness assertions.
    // In multithreaded state however, we cannot clone into `transformRegion` so
    // we build a new single-block region and clone the transform op into it.
    Region r;
    OpBuilder b(target->getContext());
    b.createBlock(&r);
    transform::TransformOptions options;
#ifndef NDEBUG
    options = options.enableExpensiveChecks();
#endif
    auto xform = cast<transform::TransformOpInterface>(b.clone(*transform));
    auto g = llvm::make_scope_exit([&]() { xform->erase(); });
    if (failed(transform::applyTransforms(target, xform, {}, options)))
      return failure();
  }
  return success();
}

/// Finds the single top-level transform operation with `root` as ancestor.
/// Reports an error if there is more than one such operation and returns the
/// first one found. Reports an error returns nullptr if no such operation
/// found.
static Operation *findTopLevelTransform(Operation *root, StringRef debugStr) {
  ::mlir::transform::TransformOpInterface topLevelTransform = nullptr;
  WalkResult walkResult = root->walk<WalkOrder::PreOrder>(
      [&](::mlir::transform::TransformOpInterface transformOp) {
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
  if (walkResult.wasInterrupted())
    return nullptr;
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
    if (!attr || attr.getValue() != tagValue)
      return WalkResult::advance();

    assert(found == nullptr && "more than one op with the same tag");
    found = op;

    // In debug mode, continue the traversal to see if the tag is not
    // duplicated. This is only necessary to ensure that the assert above is
    // triggered. In the non-debug mode, assert is not performed and we can
    // sparse some cycles by not iterating further.
#ifndef NDEBUG
    return WalkResult::advance();
#else
    return WalkResult::interrupt();
#endif // NDEBUG
  });
  return found;
}

/// Returns the ancestor of `target` that doesn't have a parent.
static Operation *getRootOperation(Operation *target) {
  Operation *root = target;
  while (root->getParentOp())
    root = root->getParentOp();
  return root;
}

/// Prints the CLI command running the repro with the current path.
static llvm::raw_ostream &
printIreeOptReproCall(llvm::raw_ostream &os, StringRef rootOpName,
                      StringRef passName,
                      const Pass::Option<std::string> &debugPayloadRootTag,
                      const Pass::Option<std::string> &debugTransformRootTag) {
  os << llvm::formatv("iree-opt --pass-pipeline=\"{0}({1}{{{2}={3} {4}={5}})\"",
                      rootOpName, passName, debugPayloadRootTag.getArgStr(),
                      debugPayloadRootTag.empty()
                          ? StringRef(kTransformIreeTagPayloadRootValue)
                          : debugPayloadRootTag,
                      debugTransformRootTag.getArgStr(),
                      debugTransformRootTag.empty()
                          ? StringRef(kTransformIreeTagTransformContainerValue)
                          : debugTransformRootTag);
  return os;
}

/// Prints the module rooted at `root` to `os` and appends
/// `transformContainer` if it is not nested in `root`.
static llvm::raw_ostream &printModuleForRepro(llvm::raw_ostream &os,
                                              Operation *root,
                                              Operation *transformContainer) {
  root->print(os);
  if (!root->isAncestor(transformContainer)) {
    transformContainer->print(os);
  }
  return os;
}

/// Saves the payload and the transform IR into a temporary file and reports
/// the file name to `os`.
static void
saveReproToTempFile(llvm::raw_ostream &os, Operation *target,
                    Operation *transformContainer, StringRef passName,
                    const Pass::Option<std::string> &debugPayloadRootTag,
                    const Pass::Option<std::string> &debugTransformRootTag) {
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
  printIreeOptReproCall(os, root->getName().getStringRef(), passName,
                        debugPayloadRootTag, debugTransformRootTag)
      << " " << filename << "\n";
  os << "===================================\n";
}

namespace {
/// Position of an op within a containing op.
struct OpPosition {
  /// Number of the containing region in the parent op.
  size_t regionNumber;
  /// Offset of the containing block in the list of blocks of the parent region.
  size_t blockOffset;
  /// Offset of the operation in the list of operations of the parent block.
  size_t opOffset;
};
} // namespace

/// Populates `positions` with the relative positions of `target` in its
/// ancestors.
static void findOpPositionInRoot(Operation *target,
                                 SmallVectorImpl<OpPosition> &positions) {
  // Root operation may or may not have a parent block and has no parent
  // region. Even if it has a parent block, we don't need its position in the
  // block because we will have a pointer to root.
  for (; target->getParentOp() != nullptr; target = target->getParentOp()) {
    size_t posInBlock =
        std::distance(target->getBlock()->begin(), target->getIterator());
    size_t blockPos = std::distance(target->getParentRegion()->begin(),
                                    target->getBlock()->getIterator());
    size_t regionNo = target->getParentRegion()->getRegionNumber();
    positions.emplace_back(OpPosition{regionNo, blockPos, posInBlock});
  }
}

/// Finds an op located at the given stack of positions (e.g., the last position
/// is at the root, and the first position is at the immediate parent) relative
/// to the given root operation. Expects the location to exist.
static Operation *getOpAtPosition(Operation *root,
                                  ArrayRef<OpPosition> positions) {
  Operation *op = root;
  for (const OpPosition &pos : llvm::reverse(positions)) {
    Region &region = op->getRegion(pos.regionNumber);
    Block &block = *std::next(region.begin(), pos.blockOffset);
    op = &*std::next(block.begin(), pos.opOffset);
  }
  return op;
}

/// Finds the clone of `original` in the cloned copy of the root operation, i.e.
/// the operation with no parent, using its relative offsets in the parent
/// parent lists of blocks and regions. Note that this performs linear
/// traversals of blocks and regions along the path to root, but is arguably
/// preferable to storing the entire mapping between all cloned operations.
static Operation *findCloned(Operation *original, Operation *cloneRoot) {
  SmallVector<OpPosition> opPositions;
  findOpPositionInRoot(original, opPositions);
  return getOpAtPosition(cloneRoot, opPositions);
}

// Optionally perform debug actions requested by the user to dump IR and a
// repro to stderr and/or a file.
static void performOptionalDebugActions(
    Operation *target, Region *transformRegion, StringRef passName,
    const Pass::Option<std::string> &debugPayloadRootTag,
    const Pass::Option<std::string> &debugTransformRootTag) {
  MLIRContext *context = target->getContext();

  // If we are not planning to print, bail before we start doing expensive
  // copying.
  bool hasDebugFlags = false;
  DEBUG_WITH_TYPE(DEBUG_TYPE_DUMP_STDERR, { hasDebugFlags = true; });
  DEBUG_WITH_TYPE(DEBUG_TYPE_DUMP_FILE, { hasDebugFlags = true; });
  if (!hasDebugFlags)
    return;

  // Since the pass may be running in parallel on multiple parts of the same
  // root operation, clone it before attaching debug attributes as it would
  // otherwise create a race. While we could avoid cloning when multithreading
  // is disabled or when the attributes are already present, this is only
  // necessary in debug builds that are not performance-critical and can afford
  // an extra copy.
  Operation *root = getRootOperation(target);
  OwningOpRef<Operation *> rootClone(root->clone());
  Operation *debugRoot = rootClone.get();
  Operation *debugTarget = findCloned(target, debugRoot);

  Operation *transformContainer = transformRegion->getParentOp();
  assert(transformContainer && "unexpected detached transform region");
  OwningOpRef<Operation *> maybeTransformContainerClone;
  Operation *debugTransformContainer;
  if (root->isAncestor(transformContainer)) {
    debugTransformContainer = findCloned(transformContainer, debugRoot);
  } else {
    maybeTransformContainerClone =
        OwningOpRef<Operation *>(transformContainer->clone());
    debugTransformContainer = maybeTransformContainerClone.get();
  }

  // Add temporary debug / repro attributes, these must never leak out.
  if (debugPayloadRootTag.empty()) {
    debugTarget->setAttr(
        kTransformIreeTagAttrName,
        StringAttr::get(context, kTransformIreeTagPayloadRootValue));
  }
  if (debugTransformRootTag.empty()) {
    debugTransformContainer->setAttr(
        kTransformIreeTagAttrName,
        StringAttr::get(context, kTransformIreeTagTransformContainerValue));
  }

  DEBUG_WITH_TYPE(DEBUG_TYPE_DUMP_STDERR, {
    llvm::dbgs() << "=== Transform Interpreter Repro ===\n";
    printIreeOptReproCall(llvm::dbgs() << "cat <<EOF | ",
                          debugRoot->getName().getStringRef(), passName,
                          debugPayloadRootTag, debugTransformRootTag)
        << "\n";
    printModuleForRepro(llvm::dbgs(), debugRoot, debugTransformContainer);
    llvm::dbgs() << "\nEOF\n";
    llvm::dbgs() << "===================================\n";
  });
  DEBUG_WITH_TYPE(DEBUG_TYPE_DUMP_FILE, {
    saveReproToTempFile(llvm::dbgs(), debugTarget, debugTransformContainer,
                        passName, debugPayloadRootTag, debugTransformRootTag);
  });
}

LogicalResult
transform::iree_dialects::detail::interpreterBaseRunOnOperationImpl(
    Operation *target, StringRef passName,
    const std::shared_ptr<OwningOpRef<ModuleOp>> &sharedTransformModule,
    const Pass::Option<std::string> &transformFileName,
    const Pass::Option<std::string> &debugPayloadRootTag,
    const Pass::Option<std::string> &debugTransformRootTag) {
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
    if (!topLevelTransform)
      return failure();
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
      return failure();
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
      return failure();
    }
    if (transformRoot->getNumRegions() != 1 ||
        !transformRoot->getRegion(0).hasOneBlock()) {
      transformRoot->emitError() << "expected transform container op to have "
                                    "one single-block region";
      return failure();
    }
    transformRegion = &transformRoot->getRegion(0);
  }

  // Step 4
  // ------
  // Optionally perform debug actions requested by the user to dump IR and a
  // repro to stderr and/or a file.
  performOptionalDebugActions(target, transformRegion, passName,
                              debugPayloadRootTag, debugTransformRootTag);

  // Step 5
  // ------
  // Apply the transform to the IR
  // TODO: lift this assertion.
  assert(transformRegion->getBlocks().size() == 1 &&
         "expected single-region block");
  if (failed(applyTransformsInRegion(*transformRegion, payloadRoot))) {
    payloadRoot->emitError() << "transform dialect interpreter failed";
    return failure();
  }

  return success();
}

LogicalResult transform::iree_dialects::detail::interpreterBaseInitializeImpl(
    MLIRContext *context, StringRef transformFileName,
    std::shared_ptr<OwningOpRef<ModuleOp>> &module) {
  OwningOpRef<ModuleOp> parsed;
  if (failed(parseTransformModuleFromFile(context, transformFileName, parsed)))
    return failure();

  module = std::make_shared<OwningOpRef<ModuleOp>>(std::move(parsed));
  return success();
}

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

//===- LaunchConfig.cpp - Specifies configuration used to drive the nfo ---===//
//
// This file defines the data structure that is used by the codegeneration to
// lower to target specific IR. The values of the parameters are archtecture
// specific. Once set the same transformations can be used to generate the
// desired code. This allows sharing codegen infra between different backends.
//
//===----------------------------------------------------------------------===//

#include "iree/compiler/Conversion/Common/LaunchConfig.h"

#include "iree/compiler/Conversion/CodegenUtils/FunctionUtils.h"
#include "iree/compiler/Conversion/Common/Attributes.h"
#include "llvm/Support/FormatVariadic.h"
#include "mlir/Dialect/Linalg/Analysis/DependenceAnalysis.h"
#include "mlir/Dialect/Linalg/IR/LinalgOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"

namespace mlir {
namespace iree_compiler {

/// Name of the StrAttr that can be used to get the key to access the tile size
/// information.
static const char kLaunchInfoKey[] = "launch_info_key";

static Optional<StringRef> getKey(Operation *op) {
  StringAttr attr = op->getAttrOfType<StringAttr>(kLaunchInfoKey);
  if (!attr) return {};
  return attr.getValue();
}

static void setKey(Operation *op, StringRef key) {
  MLIRContext *context = op->getContext();
  op->setAttr(Identifier::get(kLaunchInfoKey, op->getContext()),
              StringAttr::get(context, key));
}

static std::string getOrSetNewKey(Operation *op, int64_t suffix) {
  Optional<StringRef> key = getKey(op);
  if (key) return key->str();
  std::string newKey = llvm::formatv("__op_num_{0}__", suffix).str();
  setKey(op, StringRef(newKey));
  return newKey;
}

void LaunchConfig::finalize(FuncOp funcOp) {
  funcOp.walk([&](linalg::LinalgOp linalgOp) {
    linalgOp.removeAttr(Identifier::get(kLaunchInfoKey, funcOp.getContext()));
  });
}

TileSizesListTypeRef LaunchConfig::getTileSizes(Operation *op) const {
  auto key = getKey(op);
  if (!key) return {};
  auto it = tileSizes.find(*key);
  if (it == tileSizes.end()) return {};
  return it->second;
}

ArrayRef<int64_t> LaunchConfig::getTileSizes(Operation *op,
                                             size_t level) const {
  auto t = getTileSizes(op);
  if (level >= t.size()) return {};
  return t[level];
}

Operation *LaunchConfig::getRootOperation(ArrayRef<Operation *> ops) {
  for (auto op : ops) {
    auto key = getKey(op);
    if (key && key.getValue() == rootOperationKey) return op;
  }
  return nullptr;
}

void LaunchConfig::setTileSizes(Operation *op, TileSizesListType vTileSizes) {
  tileSizes[getOrSetNewKey(op, tileSizes.size())] = vTileSizes;
}

void LaunchConfig::setTileSizes(Operation *op, ArrayRef<int64_t> vTileSizes,
                                size_t level) {
  tileSizes[getOrSetNewKey(op, tileSizes.size())].emplace_back(
      vTileSizes.begin(), vTileSizes.end());
}

static void setArrayVals(std::array<int64_t, 3> &array,
                         ArrayRef<int64_t> vals) {
  if (vals.size() > 3) vals = vals.take_front(3);
  for (auto size : enumerate(vals)) array[size.index()] = size.value();
  for (unsigned i : llvm::seq<unsigned>(vals.size(), 3)) array[i] = 1;
}

void LaunchConfig::setWorkgroupSize(ArrayRef<int64_t> vWorkgroupSize) {
  setArrayVals(workgroupSize, vWorkgroupSize);
}

void LaunchConfig::setNumSubgroups(ArrayRef<int64_t> vNumSubgroups) {
  setArrayVals(numSubgroups, vNumSubgroups);
}

void LaunchConfig::setRootOperation(Operation *op) {
  Optional<StringRef> key = getKey(op);
  if (key) rootOperationKey = *key;
}

void LaunchConfig::setSameConfig(Operation *source, Operation *target) {
  assert(getKey(source) && "missing configuration of source operation");
  setKey(target, *getKey(source));
}

void LaunchConfig::setVectorize(bool enableVectorize) {
  vectorize = enableVectorize;
}

LogicalResult propogateRootOperationLaunchConfig(
    LaunchConfig &config, linalg::LinalgOp rootOperation,
    const linalg::LinalgDependenceGraph &dependenceGraph) {
  auto dependences = dependenceGraph.getDependentOperations(rootOperation);
  // The dependent operations get the same tile size information as the root
  // operation. To propogate that information, just use the same key as the root
  // operation.
  for (auto dependence : dependences) {
    config.setSameConfig(rootOperation, dependence.getDependentOp());
  }
  return success();
}
}  // namespace iree_compiler
}  // namespace mlir

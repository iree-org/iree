// Copyright 2019 Google LLC
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

#include "iree/compiler/Utils/IREECodegenUtils.h"

namespace mlir {
namespace iree_compiler {

/// Gets the launch size associated with the dispatch function.
LogicalResult getLegacyLaunchSize(Operation *funcOp,
                                  SmallVectorImpl<int64_t> &launchSize) {
  if (!funcOp->getAttr("iree.executable.export")) {
    return funcOp->emitError(
        "expected operation to be in dispatch function to get launch size");
  }
  auto workloadAttr =
      funcOp->getAttrOfType<DenseElementsAttr>("iree.executable.workload");
  if (!workloadAttr) {
    return funcOp->emitError(
        "unable to find workload size, missing attribute "
        "iree.executable.workload in dispatch function");
  }
  launchSize.clear();
  for (auto value : workloadAttr.getValues<APInt>()) {
    launchSize.push_back(value.getSExtValue());
  }
  // Drop trailing ones.
  auto dropFrom = launchSize.size() - 1;
  while (dropFrom > 0 && launchSize[dropFrom] == 1) {
    --dropFrom;
  }
  if (dropFrom > 0) {
    launchSize.erase(std::next(launchSize.begin(), dropFrom + 1),
                     launchSize.end());
  }
  return success();
}

/// Gets the workgroup size.
template <typename intType>
LogicalResult getLegacyWorkGroupSize(Operation *funcOp,
                                     SmallVectorImpl<intType> &workGroupSize) {
  if (!funcOp->getAttr("iree.executable.export")) {
    return funcOp->emitError(
        "expected operation to be in dispatch function to get launch size");
  }
  auto workGroupSizeAttr = funcOp->getAttrOfType<DenseElementsAttr>(
      "iree.executable.workgroup_size");
  if (!workGroupSizeAttr) {
    return funcOp->emitError(
        "unable to find workload size, missing attribute "
        "iree.executable.workload in dispatch function");
  }
  workGroupSize.clear();
  for (auto value : workGroupSizeAttr.getValues<APInt>()) {
    workGroupSize.push_back(value.getSExtValue());
  }
  return success();
}

template LogicalResult getLegacyWorkGroupSize<int32_t>(
    Operation *funcOp, SmallVectorImpl<int32_t> &workGroupSize);
template LogicalResult getLegacyWorkGroupSize<int64_t>(
    Operation *funcOp, SmallVectorImpl<int64_t> &workGroupSize);

}  // namespace iree_compiler
}  // namespace mlir

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

#include "iree/compiler/Dialect/HAL/Target/TargetBackend.h"

#include <algorithm>

#include "llvm/Support/CommandLine.h"

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace HAL {

TargetOptions getTargetOptionsFromFlags() {
  static llvm::cl::OptionCategory halTargetOptionsCategory(
      "IREE HAL executable target options");

  // This function is called as part of registering the pass
  // TranslateExecutablesPass. Pass registery is also staticly initialized,
  // so targetBackendsFlags needs to be here to be initialized first.
  static llvm::cl::list<std::string> *targetBackendsFlag =
      new llvm::cl::list<std::string>{
          "iree-hal-target-backends",
          llvm::cl::desc("Target backends for executable compilation"),
          llvm::cl::ZeroOrMore, llvm::cl::cat(halTargetOptionsCategory)};

  TargetOptions targetOptions;
  targetOptions.targets = *targetBackendsFlag;
  return targetOptions;
}

// static
bool TargetBackend::matchPattern(StringRef value, StringRef pattern) {
  size_t nextCharIndex = pattern.find_first_of("*?");
  if (nextCharIndex == std::string::npos) {
    return value == pattern;
  } else if (nextCharIndex > 0) {
    if (value.substr(0, nextCharIndex) != pattern.substr(0, nextCharIndex)) {
      return false;
    }
    value = value.substr(nextCharIndex);
    pattern = pattern.substr(nextCharIndex);
  }
  if (value.empty() && pattern.empty()) {
    return true;
  }
  char patternChar = pattern[0];
  if (patternChar == '*' && pattern.size() > 1 && value.empty()) {
    return false;
  } else if (patternChar == '*' && pattern.size() == 1) {
    return true;
  } else if (patternChar == '?' || value[0] == patternChar) {
    return matchPattern(value.substr(1), pattern.substr(1));
  } else if (patternChar == '*') {
    return matchPattern(value, pattern.substr(1)) ||
           matchPattern(value.substr(1), pattern);
  }
  return false;
}

void TargetBackend::constructTargetOps(IREE::Flow::ExecutableOp sourceOp,
                                       IREE::HAL::ExecutableOp executableOp) {
  OpBuilder targetBuilder(&executableOp.getBlock().back());
  auto targetContainerOp = targetBuilder.create<IREE::HAL::ExecutableTargetOp>(
      sourceOp.getLoc(), name());
  OpBuilder containerBuilder(&targetContainerOp.getBlock().back());
  containerBuilder.clone(*sourceOp.getInnerModule().getOperation());
}

std::array<Value, 3> TargetBackend::calculateDispatchWorkgroupSize(
    Location loc, IREE::HAL::ExecutableOp executableOp,
    IREE::HAL::ExecutableEntryPointOp entryPointOp, Value workload,
    OpBuilder &builder) {
  // When no workgroup size is specified we just assume [1,1,1].
  // This yields a workgroup count that models the extents of the workload.
  return {
      builder.createOrFold<mlir::ConstantIndexOp>(loc, 1),
      builder.createOrFold<mlir::ConstantIndexOp>(loc, 1),
      builder.createOrFold<mlir::ConstantIndexOp>(loc, 1),
  };
}

std::array<Value, 3> TargetBackend::calculateDispatchWorkgroupCount(
    Location loc, IREE::HAL::ExecutableOp executableOp,
    IREE::HAL::ExecutableEntryPointOp entryPointOp, Value workload,
    OpBuilder &builder) {
  auto workgroupSize = calculateDispatchWorkgroupSize(
      loc, executableOp, entryPointOp, workload, builder);
  return calculateDispatchWorkgroupCount(loc, workload, workgroupSize, builder);
}

std::array<Value, 3> TargetBackend::calculateDispatchWorkgroupCount(
    Location loc, Value workload, const std::array<Value, 3> &workgroupSize,
    OpBuilder &builder) {
  std::array<Value, 3> result;
  auto constantOne = builder.createOrFold<mlir::ConstantIndexOp>(loc, 1);
  for (int i = 0; i < 3; ++i) {
    // Round up: (workload + workgroup_size - 1) / workgroup_size;
    auto rounded = builder.createOrFold<mlir::SubIOp>(
        loc,
        builder.createOrFold<mlir::AddIOp>(loc, workload, workgroupSize[i]),
        constantOne);
    auto workgroupCountI = builder.createOrFold<mlir::UnsignedDivIOp>(
        loc, rounded, workgroupSize[i]);
    result[i] = workgroupCountI;

    // Multiply back out and subtract from invocations.
    workload = builder.createOrFold<SubIOp>(
        loc, workload,
        builder.createOrFold<MulIOp>(loc, workgroupCountI, rounded));

    // Ensure > 0.
    auto workloadGreaterZero =
        builder.create<CmpIOp>(loc, CmpIPredicate::sge, workload, constantOne);
    workload = builder.create<SelectOp>(loc, workloadGreaterZero, workload,
                                        constantOne);
  }
  return result;
}

LogicalResult TargetBackend::recordDispatch(
    Location loc, DispatchState dispatchState,
    DeviceSwitchBuilder &switchBuilder) {
  auto *region = switchBuilder.addConditionRegion(
      IREE::HAL::DeviceMatchIDAttr::get(name(), loc.getContext()),
      {
          dispatchState.workload,
          dispatchState.commandBuffer,
          dispatchState.executable,
      });
  auto &entryBlock = region->front();
  auto workload = entryBlock.getArgument(0);
  auto commandBuffer = entryBlock.getArgument(1);
  auto executable = entryBlock.getArgument(2);

  auto builder = OpBuilder::atBlockBegin(&entryBlock);
  auto workgroupCount = calculateDispatchWorkgroupCount(
      loc, dispatchState.executableOp, dispatchState.entryPointOp, workload,
      builder);
  builder.create<IREE::HAL::CommandBufferDispatchOp>(
      loc, commandBuffer, executable, dispatchState.entryPointOp,
      workgroupCount[0], workgroupCount[1], workgroupCount[2]);

  builder.create<IREE::HAL::ReturnOp>(loc);
  return success();
}

}  // namespace HAL
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir

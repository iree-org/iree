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

// clang-format off

// NOLINTNEXTLINE
// RUN: test-unit-features 2>&1 | IreeFileCheck %s

// clang-format on

#include "experimental/ModelBuilder/ModelBuilder.h"
#include "experimental/ModelBuilder/ModelRunner.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"

using namespace mlir;  // NOLINT
void wrapTest(std::string funcName, std::function<bool(ModelBuilder &)> test) {
  ModelBuilder modelBuilder;
  {
    auto f = modelBuilder.makeFunction(funcName, {}, {});
    OpBuilder b(&f.getBody());
    ScopedContext scope(b, f.getLoc());
    // CHECK: success
    if (test(modelBuilder)) {
      printf("success\n");
    } else {
      printf("failure\n");
    }
    std_ret();
  }
}

double extractDoubleFromConstantFloat(Value v) {
  return dyn_cast<ConstantFloatOp>(v.getDefiningOp())
      .getValue()
      .convertToDouble();
}
bool testConstructF64(ModelBuilder &modelBuilder) {
  double input = 19204.89345;
  auto f64 = modelBuilder.constant_f64(input);
  assert(f64.getType() == modelBuilder.f64);
  assert(extractDoubleFromConstantFloat(f64) == input);
  return true;
}

bool testCapturedValueHandle(ModelBuilder &modelBuilder) {
  CapturedValueHandle value(modelBuilder.f64);
  auto fvalue = value.capture(
      [&]() { return ValueHandle(modelBuilder.constant_f64(1.0)); });
  assert(extractDoubleFromConstantFloat(fvalue.getValue()) == 1.0);
  auto svalue = value.capture(
      [&]() { return ValueHandle(modelBuilder.constant_f64(2.0)); });
  assert(extractDoubleFromConstantFloat(svalue.getValue()) == 1.0);
  return true;
}

int main(int argc, char **argv) {
  // Allow LLVM setup through command line and parse the
  // test specific option for a runtime support library.
  llvm::InitLLVM y(argc, argv);
  llvm::cl::ParseCommandLineOptions(argc, argv, "TestUnitFeatures\n");

  wrapTest("testConstructF64", testConstructF64);
  wrapTest("testCapturedValueHandle", testCapturedValueHandle);
}

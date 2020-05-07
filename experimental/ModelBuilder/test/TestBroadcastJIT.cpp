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
// RUN: test-broadcast-jit -runtime-support=$(dirname %s)/runtime-support.so 2>&1 | IreeFileCheck %s

// clang-format on

#include <memory>
#include "experimental/ModelBuilder/ModelBuilder.h"
#include "experimental/ModelBuilder/ModelRunner.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"
#include "mlir/ExecutionEngine/RunnerUtils.h"

using namespace mlir;  // NOLINT

static llvm::cl::opt<std::string> runtimeSupport(
    "runtime-support", llvm::cl::desc("Runtime support library filename"),
    llvm::cl::value_desc("filename"), llvm::cl::init("-"));

namespace {

SmallVector<int64_t, 2> getMemRefLayout(ArrayRef<int64_t> shape) {
  return SmallVector<int64_t, 2>(shape.size(),
                                 MemRefType::getDynamicStrideOrOffset());
}

template <unsigned InputRank, unsigned OutputRank,
          typename FreeFunType = decltype(&::free)>
void testBroadcast(StringLiteral funcName, ArrayRef<int64_t> irInputShape,
                   ArrayRef<int64_t> irOutputShape,
                   std::unique_ptr<StridedMemRefType<float, InputRank>,
                                   FreeFunType>& inputData,
                   std::unique_ptr<StridedMemRefType<float, OutputRank>,
                                   FreeFunType>& outputData) {
  ModelBuilder modelBuilder;
  // Build IR.
  {
    using namespace mlir::edsc;
    auto f32 = modelBuilder.f32;

    auto typeA = MemRefType::get(
        irInputShape, f32,
        makeStridedLinearLayoutMap(getMemRefLayout(irInputShape),
                                   /*offset=*/0, modelBuilder.getContext()));
    auto typeB = MemRefType::get(irOutputShape, f32);
    auto f =
        modelBuilder.makeFunction(funcName, {}, {typeA, typeB},
                                  MLIRFuncOpConfig().setEmitCInterface(true));
    OpBuilder b(&f.getBody());
    ScopedContext scope(b, f.getLoc());

    Value input{f.getArgument(0)}, output{f.getArgument(1)};
    AffineExpr i, j;
    bindDims(scope.getContext(), i, j);
    StructuredIndexed inputIndexed(input), outputIndexed(output);

    SmallVector<IteratorType, 2> iterTypes(irOutputShape.size(),
                                           IteratorType::Parallel);
    makeGenericLinalgOp(
        iterTypes, {inputIndexed({i})}, {outputIndexed({i, j})},
        [](ArrayRef<BlockArgument> args) {
          assert(args.size() == 2 && "expected 2 block arguments");
          linalg_yield(ValueRange(args[0]));
        });
    std_ret();
  }

  // Compile the function and run it.
  ModelRunner runner(modelBuilder.getModuleRef());
  runner.compile(CompilationOptions(), runtimeSupport);
  auto err = runner.invoke(funcName, inputData, outputData);
  if (err) llvm_unreachable("Error running function.");

  ::impl::printMemRef(*inputData);
  ::impl::printMemRef(*outputData);
}

// Allocate data within data structures that interoperate with the MLIR ABI
// conventions used by codegen.
template <unsigned Rank, typename FreeFunType = decltype(&::free)>
std::unique_ptr<StridedMemRefType<float, Rank>, FreeFunType> MakeData(
    std::array<int64_t, Rank> shape,      //
    std::array<int64_t, Rank> strides,    //
    std::array<int64_t, Rank> dataShape,  //
    ArrayRef<float> flatData = {}) {
  auto data = makeInitializedStridedMemRefDescriptor<float, Rank>(
      dataShape, [&](unsigned idx, float* ptr) {
        *(ptr + idx) = flatData.empty() ? 0 : flatData[idx];
      });
  std::copy(strides.begin(), strides.end(), data->strides);
  std::copy(shape.begin(), shape.end(), data->sizes);
  return data;
}

}  // namespace

int main(int argc, char** argv) {
  // Allow LLVM setup through command line and parse the
  // test specific option for a runtime support library.
  llvm::InitLLVM y(argc, argv);
  llvm::cl::ParseCommandLineOptions(argc, argv, "TestBroadcastJIT\n");

  // CHECK: [1, 1, 1, 1]
  // CHECK: [2, 2, 2, 2]
  // CHECK: [3, 3, 3, 3]
  {
    auto in = MakeData<1>({3}, {1}, {3}, {1.0, 2.0, 3.0});
    auto out = MakeData<2>({3, 4}, {4, 1}, {3, 4});
    testBroadcast<1, 2>("static_trivial_broadcast", {3}, {3, 4}, in, out);
  }

  // CHECK: [1, 1, 1, 1]
  // CHECK: [1, 1, 1, 1]
  // CHECK: [1, 1, 1, 1]
  {
    auto in = MakeData<1>({3}, {0}, {1}, {1.0});
    auto out = MakeData<2>({3, 4}, {4, 1}, {3, 4});
    testBroadcast<1, 2>("static_broadcast_expansion", {3}, {3, 4}, in, out);
  }

  // CHECK: [1, 1, 1, 1]
  // CHECK: [2, 2, 2, 2]
  // CHECK: [3, 3, 3, 3]
  {
    auto in = MakeData<1>({3}, {1}, {3}, {1.0, 2.0, 3.0});
    auto out = MakeData<2>({3, 4}, {4, 1}, {3, 4});
    testBroadcast<1, 2>("dyn_trivial_broadcast", {-1}, {-1, -1}, in, out);
  }

  // CHECK: [1, 1, 1, 1]
  // CHECK: [1, 1, 1, 1]
  // CHECK: [1, 1, 1, 1]
  {
    auto in = MakeData<1>({3}, {0}, {1}, {1.0});
    auto out = MakeData<2>({3, 4}, {4, 1}, {3, 4});
    testBroadcast<1, 2>("dyn_broadcast_expansion", {-1}, {-1, -1}, in, out);
  }
}

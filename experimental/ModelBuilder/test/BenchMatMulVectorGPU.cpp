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
#include <string>

#include "experimental/ModelBuilder/ModelBuilder.h"
#include "experimental/ModelBuilder/ModelRunner.h"
#include "experimental/ModelBuilder/VulkanWrapperPass.h"
#include "iree/compiler/Conversion/CodegenUtils/ForOpCanonicalization.h"
#include "iree/compiler/Conversion/CodegenUtils/TransformUtils.h"
#include "iree/compiler/Conversion/LinalgToSPIRV/MemorySpace.h"
#include "iree/compiler/Conversion/LinalgToSPIRV/Passes.h"
#include "iree/compiler/Conversion/LinalgToSPIRV/Utils.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"
#include "mlir/Conversion/GPUToVulkan/ConvertGPUToVulkanPass.h"
#include "mlir/Conversion/LinalgToLLVM/LinalgToLLVM.h"
#include "mlir/Conversion/StandardToLLVM/ConvertStandardToLLVMPass.h"
#include "mlir/Conversion/StandardToSPIRV/StandardToSPIRVPass.h"
#include "mlir/Dialect/GPU/Passes.h"
#include "mlir/Dialect/Linalg/EDSC/Intrinsics.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/Linalg/Transforms/CodegenStrategy.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVOps.h"
#include "mlir/Dialect/SPIRV/IR/TargetAndABI.h"
#include "mlir/Dialect/SPIRV/Transforms/Passes.h"
#include "mlir/Dialect/Vector/VectorOps.h"
#include "mlir/ExecutionEngine/CRunnerUtils.h"
#include "mlir/ExecutionEngine/RunnerUtils.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/Parser.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"

using namespace mlir;                    // NOLINT
using namespace mlir::edsc;              // NOLINT
using namespace mlir::edsc::intrinsics;  // NOLINT

static llvm::cl::opt<std::string> vulkanWrapper(
    "vulkan-wrapper", llvm::cl::desc("Vulkan wrapper library"),
    llvm::cl::value_desc("filename"), llvm::cl::init("-"));

static llvm::cl::opt<bool> correctness(
    "correctness",
    llvm::cl::desc(
        "Compare the result to value calculated on CPU. We will use a smaller "
        "matrix multiply in this case to avoid long runtime."),
    llvm::cl::init(false));

static llvm::cl::opt<bool> useWorkgroupMemory(
    "use-workgroup-memory", llvm::cl::desc("Enable use of workgroup memory"),
    llvm::cl::value_desc("boolean"), llvm::cl::init(false));

static llvm::cl::opt<bool> enableLICM(
    "enable-licm",
    llvm::cl::desc("Enable loop invariant hoisting optimizations"),
    llvm::cl::value_desc("boolean"), llvm::cl::init(true));

static llvm::cl::opt<std::string> matType("matrix-type",
                                          llvm::cl::desc("Matrix element type"),
                                          llvm::cl::value_desc("type"),
                                          llvm::cl::init("i8xi8xi32"));

static llvm::cl::opt<std::string> target(
    "target", llvm::cl::desc("Platform target to decide the strategy"),
    llvm::cl::value_desc("type"), llvm::cl::init(""));

static void addLoweringPasses(mlir::PassManager &pm,
                              llvm::ArrayRef<int64_t> numWorkgroups,
                              llvm::ArrayRef<Type> args) {
  pm.addPass(mlir::iree_compiler::createVectorToGPUPass());
  pm.addPass(mlir::createLowerAffinePass());
  pm.addPass(memref::createFoldSubViewOpsPass());
  pm.addPass(mlir::createCanonicalizerPass());
  pm.addPass(mlir::createCSEPass());
  pm.addPass(mlir::iree_compiler::createVectorizeMemref());
  pm.addPass(mlir::createCanonicalizerPass());
  pm.addPass(mlir::iree_compiler::createForOpCanonicalizationPass());
  pm.addPass(mlir::createCSEPass());
  pm.addPass(mlir::iree_compiler::createConvertToSPIRVPass());

  auto &spirvModulePM = pm.nest<mlir::spirv::ModuleOp>();
  spirvModulePM.addPass(mlir::createSetSpirvABIPass());
  spirvModulePM.addPass(mlir::spirv::createLowerABIAttributesPass());
  spirvModulePM.addPass(mlir::createCanonicalizerPass());
  spirvModulePM.addPass(mlir::createCSEPass());
  spirvModulePM.addPass(
      mlir::spirv::createUpdateVersionCapabilityExtensionPass());

  pm.addPass(mlir::createAddVulkanLaunchWrapperPass(numWorkgroups, args));
  mlir::LowerToLLVMOptions llvmOptions(pm.getContext());
  llvmOptions.emitCWrappers = true;
  pm.addPass(createLowerToLLVMPass(llvmOptions));
  pm.addPass(mlir::createConvertVulkanLaunchFuncToVulkanCallsPass());
}

static void insertBarrier(OpBuilder &b, Location loc) {
  b.create<spirv::ControlBarrierOp>(loc, spirv::Scope::Workgroup,
                                    spirv::Scope::Workgroup,
                                    spirv::MemorySemantics::AcquireRelease);
}

template <typename IdOp, typename NProcsOp>
static SmallVector<linalg::ProcInfo, 2> getGpuProcIds(
    OpBuilder &b, Location loc, ArrayRef<Range> parallelLoopRanges) {
  if (parallelLoopRanges.size() != 2)
    llvm_unreachable("expected two parallel loops for matmul operation");
  Type indexType = b.getIndexType();
  SmallVector<linalg::ProcInfo, 2> procInfo(2);
  procInfo[0] = {b.create<IdOp>(loc, indexType, b.getStringAttr("y")),
                 b.create<NProcsOp>(loc, indexType, b.getStringAttr("y"))};
  procInfo[1] = {b.create<IdOp>(loc, indexType, b.getStringAttr("x")),
                 b.create<NProcsOp>(loc, indexType, b.getStringAttr("x"))};
  return procInfo;
}

constexpr int numSubgroupX = 2;
constexpr int numSubgroupY = 2;

static SmallVector<linalg::ProcInfo, 2> getSubgroupIds(
    OpBuilder &b, Location loc, ArrayRef<Range> parallelLoopRanges) {
  if (parallelLoopRanges.size() != 2)
    llvm_unreachable("expected two parallel loops for matmul operation");
  Type indexType = b.getIndexType();
  Value sg = b.create<gpu::SubgroupIdOp>(loc, indexType);
  Value vSubgroupX = b.create<ConstantIndexOp>(loc, numSubgroupX);
  Value sgdiv = b.create<SignedDivIOp>(loc, indexType, sg, vSubgroupX);
  Value vSubgroupY = b.create<ConstantIndexOp>(loc, numSubgroupY);
  SmallVector<linalg::ProcInfo, 2> procInfo(2);
  using namespace edsc::op;
  procInfo[0] = {sgdiv % vSubgroupY, vSubgroupY};
  procInfo[1] = {sg % vSubgroupX, vSubgroupX};
  return procInfo;
}

struct MatMulF32 {
  using Type = float;
  static mlir::Type getMLIRType(MLIRContext &ctx) {
    return FloatType::getF32(&ctx);
  }
};

struct MatMulI8 {
  using Type = uint8_t;
  static mlir::Type getMLIRType(MLIRContext &ctx) {
    return IntegerType::get(&ctx, 8);
  }
};

struct MatMulI32 {
  using Type = uint32_t;
  static mlir::Type getMLIRType(MLIRContext &ctx) {
    return IntegerType::get(&ctx, 32);
  }
};

// Class to emulate half float on CPU.
class fp16 {
 public:
  void fromFloat(const float &x) {
    uint32_t asInt = *(uint32_t *)&x;
    int sign = (asInt & 0x80000000) >> 31;
    int exp = ((asInt & 0x7f800000) >> 23) - 127 + 15;
    int mantissa = (asInt & 0x7FFFFF);
    if (exp > 31) exp = 31;
    if (exp < 0) exp = 0;
    sign = sign << 15;
    exp = exp << 10;
    mantissa = mantissa >> (23 - 10);
    asInt = sign | exp | mantissa;
    value = asInt;
  }
  fp16(const float &x) { fromFloat(x); }
  fp16 &operator=(const float &x) {
    fromFloat(x);
    return *this;
  }
  fp16 &operator=(const int &x) {
    fromFloat((float)x);
    return *this;
  }
  fp16 &operator+=(const fp16 &x) {
    fromFloat(toFloat() + x.toFloat());
    return *this;
  }
  float toFloat() const {
    uint32_t asInt = value;
    int sign = (asInt & 0x8000) >> 15;
    int exp = ((asInt & 0x7c00) >> 10);
    int mantissa = (asInt & 0x3FF);
    sign = sign << 31;
    if (exp > 0) {
      exp = (exp + 127 - 15) << 23;
      mantissa = mantissa << (23 - 10);
    } else {
      mantissa = 0;
    }
    asInt = sign | exp | mantissa;
    return *(float *)&asInt;
  }
  operator float() { return toFloat(); }

 private:
  uint16_t value;
};

struct MatMulF16 {
  using Type = fp16;
  static mlir::Type getMLIRType(MLIRContext &ctx) {
    return FloatType::getF16(&ctx);
  }
};

/// Functions to initialize matrix based on the type.
template <typename T>
static T getMatA(unsigned idx) {
  if (std::is_same<T, float>::value || std::is_same<T, fp16>::value)
    return ((float)(idx % 5) - 1.0f) / 2.0f;
  else
    return (3 * idx + 1) % 117;
}

template <typename T>
static T getMatB(unsigned idx) {
  if (std::is_same<T, float>::value || std::is_same<T, fp16>::value)
    return ((float)(idx % 7) - 1.0f) / 2.0f;
  else
    return idx % 127;
}

template <typename T>
static bool EqualOrClose(T a, T b) {
  if (std::is_same<T, float>::value || std::is_same<T, fp16>::value)
    return fabs((float)a - (float)b) < 0.001f;
  return a == b;
}

static linalg::CodegenStrategy createPowerVRStrategy(int tileM, int tileN,
                                                     int tileK, int warpSize) {
  const std::array<int64_t, 3> nativeSize = {1, 1, 1};
  linalg::LinalgLoopDistributionOptions WIDistribute;
  linalg::LinalgLoopDistributionOptions WGDistribute;
  WGDistribute.distributionMethod = {
      linalg::DistributionMethod::CyclicNumProcsEqNumIters,
      linalg::DistributionMethod::CyclicNumProcsEqNumIters};
  WGDistribute.procInfo = getGpuProcIds<gpu::BlockIdOp, gpu::GridDimOp>;

  WIDistribute.distributionMethod = {
      linalg::DistributionMethod::CyclicNumProcsEqNumIters,
      linalg::DistributionMethod::CyclicNumProcsEqNumIters};
  WIDistribute.procInfo = [warpSize](OpBuilder &b, Location loc,
                                     ArrayRef<Range> parallelLoopRanges) {
    Type indexType = b.getIndexType();
    SmallVector<linalg::ProcInfo, 2> procInfo(2);
    procInfo[0] = {
        b.create<gpu::ThreadIdOp>(loc, indexType, b.getStringAttr("x")),
        b.create<ConstantIndexOp>(loc, warpSize)};
    procInfo[1] = {b.create<ConstantIndexOp>(loc, 0),
                   b.create<ConstantIndexOp>(loc, 1)};
    return procInfo;
  };
  linalg::CodegenStrategy strategy;
  SmallVector<int64_t, 2> promotionList;
  // promote matrix B
  promotionList.push_back(1);
  strategy
      .tile<linalg::MatmulOp>(
          linalg::LinalgTilingOptions()
              .setLoopType(linalg::LinalgTilingLoopType::ParallelLoops)
              .setTileSizes({tileM, tileN, tileK})
              .setDistributionOptions(WGDistribute))
      .setEnableLICM(enableLICM);
  if (useWorkgroupMemory) {
    strategy.promote<linalg::MatmulOp>(
        linalg::LinalgPromotionOptions()
            .setAllocationDeallocationFns(
                mlir::iree_compiler::allocateWorkgroupMemory,
                mlir::iree_compiler::deallocateWorkgroupMemory)
            .setCopyInOutFns(mlir::iree_compiler::copyToWorkgroupMemory,
                             mlir::iree_compiler::copyToWorkgroupMemory)
            .setOperandsToPromote(promotionList)
            .setUseFullTileBuffers({false, false}));
  }
  strategy.tile<linalg::MatmulOp>(
      linalg::LinalgTilingOptions()
          .setLoopType(linalg::LinalgTilingLoopType::ParallelLoops)
          .setTileSizes({1, tileN, tileK})
          .setDistributionOptions(WIDistribute));
  strategy.vectorize<linalg::MatmulOp>()
      // TODO: Upstream to core.
      // .unrollVector<vector::ContractionOp>(nativeSize)
      ;
  (void)nativeSize;
  return strategy;
}

static linalg::CodegenStrategy createMaliStrategy(int tileM, int tileN,
                                                  int tileK, int warpSize) {
  const std::array<int64_t, 3> nativeSize = {1, 4, 1};
  linalg::LinalgLoopDistributionOptions WIDistribute;
  linalg::LinalgLoopDistributionOptions WGDistribute;
  WGDistribute.distributionMethod = {
      linalg::DistributionMethod::CyclicNumProcsEqNumIters,
      linalg::DistributionMethod::CyclicNumProcsEqNumIters};
  WGDistribute.procInfo = getGpuProcIds<gpu::BlockIdOp, gpu::GridDimOp>;

  WIDistribute.distributionMethod = {
      linalg::DistributionMethod::CyclicNumProcsEqNumIters,
      linalg::DistributionMethod::CyclicNumProcsEqNumIters};
  WIDistribute.procInfo = [warpSize](OpBuilder &b, Location loc,
                                     ArrayRef<Range> parallelLoopRanges) {
    Type indexType = b.getIndexType();
    SmallVector<linalg::ProcInfo, 2> procInfo(2);
    procInfo[1] = {
        b.create<gpu::ThreadIdOp>(loc, indexType, b.getStringAttr("x")),
        b.create<ConstantIndexOp>(loc, warpSize)};
    procInfo[0] = {b.create<ConstantIndexOp>(loc, 0),
                   b.create<ConstantIndexOp>(loc, 1)};
    return procInfo;
  };
  linalg::CodegenStrategy strategy;
  strategy
      .tile<linalg::MatmulOp>(
          linalg::LinalgTilingOptions()
              .setLoopType(linalg::LinalgTilingLoopType::ParallelLoops)
              .setTileSizes({tileM, tileN, tileK})
              .setDistributionOptions(WGDistribute))
      .setEnableLICM(enableLICM);
  strategy.tile<linalg::MatmulOp>(
      linalg::LinalgTilingOptions()
          .setLoopType(linalg::LinalgTilingLoopType::ParallelLoops)
          .setTileSizes({tileM, tileN / warpSize, tileK})
          .setDistributionOptions(WIDistribute));
  strategy.vectorize<linalg::MatmulOp>()
      // TODO: Upstream to core.
      // .unrollVector<vector::TransferReadOp>({1, 4})
      // .unrollVector<vector::ContractionOp>(nativeSize)
      ;
  (void)nativeSize;
  return strategy;
}

static linalg::CodegenStrategy createTuringStrategy(int tileM, int tileN,
                                                    int tileK) {
  std::array<int64_t, 3> nativeSize;
  if (matType == "i8xi8xi32")
    nativeSize = {16, 16, 32};
  else if (matType == "f16xf16xf16")
    nativeSize = {16, 16, 16};
  else if (matType == "f16xf16xf32")
    nativeSize = {16, 16, 16};
  else
    llvm::errs() << "unsupported matrix type";
  linalg::LinalgLoopDistributionOptions WGDistribute;
  WGDistribute.distributionMethod = {
      linalg::DistributionMethod::CyclicNumProcsEqNumIters,
      linalg::DistributionMethod::CyclicNumProcsEqNumIters};
  WGDistribute.procInfo = getGpuProcIds<gpu::BlockIdOp, gpu::GridDimOp>;

  linalg::LinalgLoopDistributionOptions SGDistribute;
  SGDistribute.distributionMethod = {
      linalg::DistributionMethod::CyclicNumProcsEqNumIters,
      linalg::DistributionMethod::CyclicNumProcsEqNumIters};
  SGDistribute.procInfo = getSubgroupIds;

  linalg::CodegenStrategy strategy;
  strategy
      .tile<linalg::MatmulOp>(
          linalg::LinalgTilingOptions()
              .setLoopType(linalg::LinalgTilingLoopType::ParallelLoops)
              .setTileSizes({tileM, tileN, tileK})
              .setDistributionOptions(WGDistribute))
      .setEnableLICM(enableLICM);
  if (useWorkgroupMemory) {
    strategy
        .promote<linalg::MatmulOp>(
            linalg::LinalgPromotionOptions()
                .setAllocationDeallocationFns(
                    mlir::iree_compiler::allocateWorkgroupMemory,
                    mlir::iree_compiler::deallocateWorkgroupMemory)
                .setCopyInOutFns(mlir::iree_compiler::copyToWorkgroupMemory,
                                 mlir::iree_compiler::copyToWorkgroupMemory)
                .setOperandsToPromote({0, 1})
                .setUseFullTileBuffers({false, false}))
        .tile<linalg::MatmulOp>(
            linalg::LinalgTilingOptions()
                .setLoopType(linalg::LinalgTilingLoopType::ParallelLoops)
                .setTileSizes(
                    {tileM / numSubgroupY, tileN / numSubgroupX, tileK})
                .setDistributionOptions(SGDistribute));
  }
  strategy.vectorize<linalg::MatmulOp>()
      // TODO: Upstream to core.
      // .unrollVector<vector::ContractionOp>(nativeSize)
      ;
  (void)nativeSize;
  return strategy;
}

template <typename SrcT, typename DstT>
static void matMul(int m, int n, int k, int tileM, int tileN, int tileK,
                   bool correctness, int warpSize) {
  const int resRows = m;
  const int resColumns = n;
  const int reductionSize = k;
  StringLiteral funcName = "kernel_matmul";
  ModelBuilder modelBuilder;
  MLIRContext &ctx = *modelBuilder.getContext();
  auto typeA = modelBuilder.getMemRefType({resRows, reductionSize},
                                          SrcT::getMLIRType(ctx));
  auto typeB = modelBuilder.getMemRefType({reductionSize, resColumns},
                                          SrcT::getMLIRType(ctx));
  auto typeC =
      modelBuilder.getMemRefType({resRows, resColumns}, DstT::getMLIRType(ctx));
  // 1. Build the kernel.
  {
    modelBuilder.addGPUAttr();
    FuncOp kernelFunc = modelBuilder.makeFunction(
        funcName, {}, {typeA, typeB, typeC}, MLIRFuncOpConfig());
    int workgroupSize;
    if (useWorkgroupMemory)
      workgroupSize = warpSize * numSubgroupX * numSubgroupY;
    else
      workgroupSize = warpSize;
    // Right now we map one workgroup to one warp.
    kernelFunc->setAttr(
        spirv::getEntryPointABIAttrName(),
        spirv::getEntryPointABIAttr({workgroupSize, 1, 1}, &ctx));
    OpBuilder b(&kernelFunc.getBody());
    ScopedContext scope(b, kernelFunc.getLoc());

    auto A = kernelFunc.getArgument(0);
    auto B = kernelFunc.getArgument(1);
    auto C = kernelFunc.getArgument(2);

    linalg_matmul(ValueRange{A, B}, ValueRange{C});
    std_ret();
  }

  // 2. Compile the function, pass in runtime support library to the execution
  // engine for vector.print.
  ModelRunner runner(modelBuilder.getModuleRef(),
                     ModelRunner::Target::GPUTarget);
  CompilationOptions options;
  options.loweringPasses = [&](mlir::PassManager &pm) {
    linalg::CodegenStrategy strategy;

    if (target == "powerVR") {
      strategy = createPowerVRStrategy(tileM, tileN, tileK, warpSize);
    } else if (target == "NVTuring") {
      strategy = createTuringStrategy(tileM, tileN, tileK);
    } else if (target == "mali") {
      strategy = createMaliStrategy(tileM, tileN, tileK, warpSize);
    }
    modelBuilder.getModuleRef()->walk(
        [&](FuncOp fn) { strategy.transform(fn); });
    addLoweringPasses(pm, {resColumns / tileN, resRows / tileM, 1},
                      {typeA, typeB, typeC});
  };
  runner.compile(options, {vulkanWrapper});

  // 3. Allocate data within data structures that interoperate with the MLIR ABI
  // conventions used by codegen.
  auto initA = [](unsigned idx, typename SrcT::Type *ptr) {
    ptr[idx] = getMatA<typename SrcT::Type>(idx);
  };
  auto initB = [](unsigned idx, typename SrcT::Type *ptr) {
    ptr[idx] = getMatB<typename SrcT::Type>(idx);
  };
  auto zeroInit = [](unsigned idx, typename DstT::Type *ptr) { ptr[idx] = 0; };
  auto A = makeInitializedStridedMemRefDescriptor<typename SrcT::Type, 2>(
      {resRows, reductionSize}, initA);
  auto B = makeInitializedStridedMemRefDescriptor<typename SrcT::Type, 2>(
      {reductionSize, resColumns}, initB);
  auto C = makeInitializedStridedMemRefDescriptor<typename DstT::Type, 2>(
      {resRows, resColumns}, zeroInit);
  auto CPURes = makeInitializedStridedMemRefDescriptor<typename DstT::Type, 2>(
      {resRows, resColumns}, zeroInit);

  // Is checking corretness compare to the value computed on CPU.
  if (correctness) {
    for (int i = 0; i < resRows; i++) {
      for (int j = 0; j < resColumns; j++) {
        typename DstT::Type acc = (*C)[i][j];
        for (int k = 0; k < reductionSize; k++) {
          typename DstT::Type a = (*A)[i][k];
          typename DstT::Type b = (*B)[k][j];
          acc += a * b;
        }
        (*CPURes)[i][j] = acc;
      }
    }
  }

  // 4. Call the funcOp named `funcName`.
  auto err = runner.invoke(std::string(funcName) + "_wrapper", A, B, C);
  if (err) llvm_unreachable("Error running function.");

  if (correctness) {
    bool correct = true;
    for (int i = 0; i < resRows; i++) {
      for (int j = 0; j < resColumns; j++) {
        if (!EqualOrClose((*CPURes)[i][j], (*C)[i][j])) {
          correct = false;
          llvm::errs() << "mismatch at index(" << i << ", " << j
                       << ") was expecting " << (*CPURes)[i][j] << " but got "
                       << (*C)[i][j] << "\n";
        }
      }
    }
    if (correct) printf("pass\n");
  }
}

static void matMul(int m, int n, int k, int tileM, int tileN, int tileK,
                   bool correctness, int warpSize) {
  if (matType == "i8xi8xi32") {
    return matMul<MatMulI8, MatMulI32>(m, n, k, tileM, tileN, tileK,
                                       correctness, warpSize);
  }
  if (matType == "f16xf16xf16") {
    return matMul<MatMulF16, MatMulF16>(m, n, k, tileM, tileN, tileK,
                                        correctness, warpSize);
  }
  if (matType == "f16xf16xf32") {
    return matMul<MatMulF16, MatMulF32>(m, n, k, tileM, tileN, tileK,
                                        correctness, warpSize);
  }
  if (matType == "f32xf32xf32") {
    return matMul<MatMulF32, MatMulF32>(m, n, k, tileM, tileN, tileK,
                                        correctness, warpSize);
  }
  llvm_unreachable("Unsupported matrix type");
}

int main(int argc, char **argv) {
  // Allow LLVM setup through command line and parse the
  // test specific option for a runtime support library.
  llvm::InitLLVM y(argc, argv);
  llvm::cl::ParseCommandLineOptions(argc, argv, "BenchMatMulVectorGPU\n");
  if (target.empty()) {
    llvm::errs() << "No target specified.";
    return 0;
  }
  int m = 4096;
  int n = 4096;
  int k = 4096;
  if (correctness) {
    m = 256;
    n = 256;
    k = 256;
  }
  int warpSize = 32;
  std::pair<int, int> tileMRange;
  std::pair<int, int> tileNRange;
  std::pair<int, int> tileKRange;
  if (target == "powerVR") {
    m = std::min(m, 1024);
    n = std::min(n, 1024);
    k = std::min(k, 1024);
    tileMRange = {32, 32};
    tileNRange = {32, 32};
    tileKRange = {4, 4};
  } else if (target == "NVTuring") {
    tileMRange = {32, 256};
    tileNRange = {32, 256};
    tileKRange = {32, 64};
    // Workgroup memory requires at least a tile size of 128x128 to be able
    // to do full speed copy from video memory to shared local memory.
    if (useWorkgroupMemory) {
      tileMRange.first = 128;
      tileNRange.first = 128;
    }
  } else if (target == "mali") {
    warpSize = 16;
    tileMRange = {1, 8};
    tileNRange = {64, 128};
    tileKRange = {4, 4};
  } else {
    llvm::errs() << "Unknown target";
    return 0;
  }

  printf("Matrix size: %ix%ix%i\n", m, n, k);
  for (int tileK = tileKRange.first; tileK <= tileKRange.second; tileK *= 2) {
    for (int tileM = tileMRange.first; tileM <= tileMRange.second; tileM *= 2) {
      for (int tileN = tileNRange.first; tileN <= tileNRange.second;
           tileN *= 2) {
        printf("tileM=%i tileN=%i tileK=%i\n", tileM, tileN, tileK);
        // For non-power of two tile sizes, round up the matrix size to
        // be an even multiple of the tile size.
        // TODO(thomasraoux): enable non power of two tiles once affine.min
        // folding is fixed.
        auto paddedM = (m + tileM - 1) / tileM * tileM;
        auto paddedN = (n + tileN - 1) / tileN * tileN;
        auto paddedK = (k + tileK - 1) / tileK * tileK;

        matMul(paddedM, paddedN, paddedK, tileM, tileN, tileK, correctness,
               warpSize);
      }
    }
  }
}

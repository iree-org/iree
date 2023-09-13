#include "iree/compiler/Codegen/Dialect/IREECodegenAttrs.h"
#include "iree/compiler/Codegen/LLVMCPU/PassDetail.h"
#include "iree/compiler/Codegen/LLVMCPU/Passes.h"
#include "iree/compiler/Codegen/Utils/Utils.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/Vector/Transforms/LoweringPatterns.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define DEBUG_TYPE "iree-llvmcpu-set-special-tiling-configs"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

namespace mlir {
namespace iree_compiler {
namespace {

static void setTileSizes(linalg::GenericOp intMatmul,
                         linalg::GenericOp reassociation,
                         func::FuncOp entryPointFn,
                         IREE::HAL::ExecutableTargetAttr target) {
  int mDistSize = 1;
  int nDistSize = 128;
  int mSize = 1;
  int nSize = 4;
  int kSize = 8;
  int groupSize = 1;
  SmallVector<int> mDims;
  SmallVector<int> nDims;
  SmallVector<int> kDims;
  SmallVector<int> groupDims;
  SmallVector<AffineMap> maps = intMatmul.getIndexingMapsArray();
  int lhs = 0;
  int rhs = 1;
  int out = 2;
  auto hasDim = [&](int mapIdx, int dimIdx) -> bool {
    return llvm::any_of(maps[mapIdx].getResults(), [&](AffineExpr res) {
      auto expr = res.dyn_cast<AffineDimExpr>();
      return expr && expr.getPosition() == dimIdx;
    });
  };
  for (int dim = 0; dim < intMatmul.getNumLoops(); dim++) {
    if (hasDim(lhs, dim) && hasDim(rhs, dim) && hasDim(out, dim)) {
      groupDims.push_back(dim);
    } else if (hasDim(lhs, dim) && hasDim(rhs, dim) && !hasDim(out, dim)) {
      kDims.push_back(dim);
    } else if (hasDim(lhs, dim) && !hasDim(rhs, dim) && hasDim(out, dim)) {
      mDims.push_back(dim);
    } else if (!hasDim(lhs, dim) && hasDim(rhs, dim) && hasDim(out, dim)) {
      nDims.push_back(dim);
    }
  }
  if (hasFeature(target, "+avx512bw") || hasFeature(target, "+avx512vnni")) {
    kSize = 16;
  }

  if (mDims.size() > 1 || nDims.size() > 1 || kDims.size() != 1 ||
      kDims[0] != intMatmul.getNumLoops() - 1) {
    return;
  }

  SmallVector<int64_t> distTileSizes_mm(intMatmul.getNumLoops(), 0);
  SmallVector<int64_t> parallelTileSizes_mm(intMatmul.getNumLoops(), 0);
  SmallVector<int64_t> reductionTileSizes_mm(intMatmul.getNumLoops(), 0);
  SmallVector<int64_t> lastTileSizes_mm(intMatmul.getNumLoops(), 0);

  SmallVector<int64_t> distTileSizes_re(reassociation.getNumLoops(), 0);
  SmallVector<int64_t> parallelTileSizes_re(reassociation.getNumLoops(), 0);
  SmallVector<int64_t> reductionTileSizes_re(reassociation.getNumLoops(), 0);
  SmallVector<int64_t> lastTileSizes_re(reassociation.getNumLoops(), 0);

  for (int mDim : mDims) {
    distTileSizes_mm[mDim] = mDistSize;
    parallelTileSizes_mm[mDim] = mSize;
    reductionTileSizes_mm[mDim] = mSize;

    distTileSizes_re[mDim] = mDistSize;
    parallelTileSizes_re[mDim] = mSize;
  }
  for (int nDim : nDims) {
    distTileSizes_mm[nDim] = nDistSize;
    parallelTileSizes_mm[nDim] = nSize;
    reductionTileSizes_mm[nDim] = nSize;

    distTileSizes_re[nDim] = nDistSize;
    parallelTileSizes_re[nDim] = nSize;
  }
  for (int kDim : kDims) {
    reductionTileSizes_mm[kDim] = kSize;
  }
  for (int groupDim : groupDims) {
    reductionTileSizes_mm[groupDim] = groupSize;
  }

  TileSizesListType tileSizes_mm;
  tileSizes_mm.push_back(distTileSizes_mm);
  tileSizes_mm.push_back(parallelTileSizes_mm);
  tileSizes_mm.push_back(reductionTileSizes_mm);
  tileSizes_mm.push_back(lastTileSizes_mm);

  TileSizesListType tileSizes_re;
  tileSizes_re.push_back(distTileSizes_re);
  tileSizes_re.push_back(parallelTileSizes_re);
  tileSizes_re.push_back(reductionTileSizes_re);
  tileSizes_re.push_back(lastTileSizes_re);

  IREE::Codegen::DispatchLoweringPassPipeline passPipeline =
      IREE::Codegen::DispatchLoweringPassPipeline::CPUDoubleTilingExpert;

  MLIRContext *context = entryPointFn.getContext();
  auto config_mm =
      IREE::Codegen::LoweringConfigAttr::get(context, tileSizes_mm);
  intMatmul->setAttr("lowering_config", config_mm);

  auto config_re =
      IREE::Codegen::LoweringConfigAttr::get(context, tileSizes_re);
  auto translationInfo_re = IREE::Codegen::TranslationInfoAttr::get(
      entryPointFn.getContext(), passPipeline, 0, 1);
  auto compilationInfo_re = IREE::Codegen::CompilationInfoAttr::get(
      context, config_re, translationInfo_re, ArrayRef<int64_t>({}),
      std::nullopt);

  reassociation->setAttr("compilation_info", compilationInfo_re);

  return;
}

static bool isIntegerMatmul(linalg::GenericOp genericOp) {
  if (genericOp.getNumDpsInits() != 1) {
    LDBG("Wrong number of outputs for matmul: " << genericOp.getNumDpsInits()
                                                << "\n");
    return false;
  }
  if (genericOp.getNumDpsInputs() != 2) {
    LDBG("Wrong number of inputs for matmul: " << genericOp.getNumDpsInputs()
                                               << "\n");
    return false;
  }

  unsigned numLoops = genericOp.getNumLoops();
  unsigned numReductionLoops = genericOp.getNumReductionLoops();
  if (numLoops != 3) {
    LDBG("Wrong number of loops for matmul: " << numLoops << "\n");
    return false;
  }
  if (numReductionLoops != 1) {
    LDBG("Wrong number of reduction loops for matmul: " << numReductionLoops
                                                        << "\n");
    return false;
  }
  // Work back from linalg.yield and check body of genericOp.
  auto yieldOp = cast<linalg::YieldOp>(genericOp.getBody()->getTerminator());
  Value producerOutput;
  Operation *producer;
  Operation *mulRhsProducer;

  // Producer of linalg.yield op is arith.addi
  {
    producerOutput = yieldOp->getOperand(0);
    producer = producerOutput.getDefiningOp();
    if (!producer || producer->getNumOperands() == 0)
      return false;
    if (!matchPattern(producer, m_Op<arith::AddIOp>()))
      return false;
  }

  // Producer of arith.addi op is arith.muli
  {
    producerOutput = producer->getOperand(0);
    producer = producerOutput.getDefiningOp();
    if (!producer || producer->getNumOperands() == 0)
      return false;
    if (!matchPattern(producer, m_Op<arith::MulIOp>()))
      return false;
  }

  // Producer of arith.muli op RHS is arith.extui
  {
    producerOutput = producer->getOperand(1);
    mulRhsProducer = producerOutput.getDefiningOp();
    if (!mulRhsProducer || mulRhsProducer->getNumOperands() == 0)
      return false;
    if (!matchPattern(mulRhsProducer, m_Op<arith::ExtUIOp>()))
      return false;
  }

  // Producer of arith.subf op LHS is arith.extsi
  {
    producerOutput = producer->getOperand(0);
    producer = producerOutput.getDefiningOp();
    if (!producer || producer->getNumOperands() == 0)
      return false;
    if (!matchPattern(producer, m_Op<arith::ExtSIOp>()))
      return false;
  }

  return true;
}

static bool isReassociatedDequantizationOp(linalg::GenericOp genericOp) {
  if (genericOp.getNumDpsInits() != 1) {
    LDBG("Wrong number of outputs: " << genericOp.getNumDpsInits() << "\n");
    return false;
  }
  if (genericOp.getNumDpsInputs() != 5) {
    LDBG("Wrong number of inputs: " << genericOp.getNumDpsInputs() << "\n");
    return false;
  }

  unsigned numLoops = genericOp.getNumLoops();
  unsigned numReductionLoops = genericOp.getNumReductionLoops();
  if (numLoops != 2) {
    LDBG("Wrong number of loops: " << numLoops << "\n");
    return false;
  }
  if (numReductionLoops != 1) {
    LDBG("Wrong number of reduction loops: " << numReductionLoops << "\n");
    return false;
  }
  // Work back from linalg.yield and check body of genericOp.
  auto yieldOp = cast<linalg::YieldOp>(genericOp.getBody()->getTerminator());
  Value producerOutput;
  Operation *producer;
  Operation *subRhsProducer;

  // Producer of linalg.yield op is arith.addf
  {
    producerOutput = yieldOp->getOperand(0);
    producer = producerOutput.getDefiningOp();
    if (!producer || producer->getNumOperands() == 0)
      return false;
    if (!matchPattern(producer, m_Op<arith::AddFOp>()))
      return false;
  }

  // Producer of arith.addf op is arith.subf
  {
    producerOutput = producer->getOperand(0);
    producer = producerOutput.getDefiningOp();
    if (!producer || producer->getNumOperands() == 0)
      return false;
    if (!matchPattern(producer, m_Op<arith::SubFOp>()))
      return false;
  }

  // Producer of arith.subf op RHS is arith.mulf
  {
    producerOutput = producer->getOperand(1);
    subRhsProducer = producerOutput.getDefiningOp();
    if (!subRhsProducer || subRhsProducer->getNumOperands() == 0)
      return false;
    if (!matchPattern(subRhsProducer, m_Op<arith::MulFOp>()))
      return false;
  }

  // Producer of arith.mulf from arith.subf RHS is arith.mulf
  {
    producerOutput = subRhsProducer->getOperand(0);
    subRhsProducer = producerOutput.getDefiningOp();
    if (!subRhsProducer || subRhsProducer->getNumOperands() == 0)
      return false;
    if (!matchPattern(subRhsProducer, m_Op<arith::MulFOp>()))
      return false;
  }

  // Producer of arith.subf op LHS is arith.mulf
  {
    producerOutput = producer->getOperand(0);
    producer = producerOutput.getDefiningOp();
    if (!producer || producer->getNumOperands() == 0)
      return false;
    if (!matchPattern(producer, m_Op<arith::MulFOp>()))
      return false;
  }

  // Producer of arith.mulf op is arith.mulf
  {
    producerOutput = producer->getOperand(0);
    producer = producerOutput.getDefiningOp();
    if (!producer || producer->getNumOperands() == 0)
      return false;
    if (!matchPattern(producer, m_Op<arith::MulFOp>()))
      return false;
  }

  // Producer of arith.mulf op is arith.sitofp
  {
    producerOutput = producer->getOperand(0);
    producer = producerOutput.getDefiningOp();
    if (!producer || producer->getNumOperands() == 0)
      return false;
    if (!matchPattern(producer, m_Op<arith::SIToFPOp>()))
      return false;
  }

  return true;
}

struct SetSpecialTilingConfigsPass
    : public SetSpecialTilingConfigsBase<SetSpecialTilingConfigsPass> {

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<linalg::LinalgDialect>();
  }

  void runOnOperation() override {
    auto funcOp = getOperation();
    auto target = IREE::HAL::ExecutableTargetAttr::lookup(funcOp);
    std::optional<std::pair<linalg::GenericOp, linalg::GenericOp>>
        reassociatedQuantizedMatmulOps = std::nullopt;
    for (auto genericOp :
         funcOp.getFunctionBody().getOps<linalg::GenericOp>()) {
      if (isReassociatedDequantizationOp(genericOp)) {
        auto intMatmulOp =
            genericOp.getInputs()[0].getDefiningOp<linalg::GenericOp>();
        if (intMatmulOp) {
          if (isIntegerMatmul(intMatmulOp)) {
            reassociatedQuantizedMatmulOps =
                std::make_pair(intMatmulOp, genericOp);
            break;
          }
        }
      }
    }
    if (reassociatedQuantizedMatmulOps) {
      setTileSizes(reassociatedQuantizedMatmulOps->first,
                   reassociatedQuantizedMatmulOps->second, funcOp, target);
    }
  }
};
} // namespace

std::unique_ptr<OperationPass<func::FuncOp>>
createSetSpecialTilingConfigsPass() {
  return std::make_unique<SetSpecialTilingConfigsPass>();
}

} // namespace iree_compiler
} // namespace mlir

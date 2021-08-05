#include "iree/compiler/Dialect/Flow/Utils/DispatchUtils.h"

#include "iree/compiler/Dialect/LinalgExt/IR/LinalgExtOps.h"
#include "mlir/Dialect/Linalg/IR/LinalgOps.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace Flow {

static unsigned kNumMaxParallelDims = 3;

int getNumMaxParallelDims() { return kNumMaxParallelDims; }

SmallVector<unsigned> getPartitionedLoops(Operation *op) {
  SmallVector<unsigned> partitionedLoops;
  if (auto mmt4dOp = dyn_cast<linalg::Mmt4DOp>(op)) {
    return {0, 1};
  }
  if (auto linalgOp = dyn_cast<linalg::LinalgOp>(op)) {
    auto getNumOuterParallelLoops = [](linalg::LinalgOp op) -> size_t {
      return op.iterator_types()
          .getValue()
          .take_while([](Attribute attr) -> bool {
            return linalg::isParallelIteratorType(attr);
          })
          .size();
    };
    size_t numOuterParallelLoops = getNumOuterParallelLoops(linalgOp);
    partitionedLoops =
        llvm::to_vector<4>(llvm::seq<unsigned>(0, numOuterParallelLoops));
    if (partitionedLoops.size() > kNumMaxParallelDims) {
      partitionedLoops.erase(
          partitionedLoops.begin(),
          std::next(partitionedLoops.begin(),
                    numOuterParallelLoops - kNumMaxParallelDims));
    }
    return partitionedLoops;
  }
  if (auto tilableOp = dyn_cast<linalg_ext::TiledOpInterface>(op)) {
    return tilableOp.getPartitionableLoops(kNumMaxParallelDims);
  }
  return {};
}

}  // namespace Flow
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir

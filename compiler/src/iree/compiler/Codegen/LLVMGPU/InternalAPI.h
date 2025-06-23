#ifndef IREE_COMPILER_CODEGEN_LLVMGPU_INTERNALAPI_H_
#define IREE_COMPILER_CODEGEN_LLVMGPU_INTERNALAPI_H_

#include <tuple>

#include "llvm/Support/LogicalResult.h"

namespace mlir {

namespace linalg {
class LinalgOp;
} // namespace linalg

namespace vector {
class ContractionOp;
} // namespace vector

namespace iree_compiler {
class VectorContractOpInfo;

namespace IREE {

namespace VectorExt {
class VectorLayoutInterface;
} // namespace VectorExt

namespace GPU {
  class MMAScheduleAttr;

  ::llvm::FailureOr<::std::tuple<VectorExt::VectorLayoutInterface,
                                 VectorExt::VectorLayoutInterface,
                                 VectorExt::VectorLayoutInterface>>
  getContractionLayout(IREE::GPU::MMAScheduleAttr scheduleAttr,
                       VectorContractOpInfo &opInfo,
                       linalg::LinalgOp contractOp);

  ::llvm::FailureOr<::std::tuple<VectorExt::VectorLayoutInterface,
                                 VectorExt::VectorLayoutInterface,
                                 VectorExt::VectorLayoutInterface>>
  getContractionLayout(IREE::GPU::MMAScheduleAttr scheduleAttr,
                       VectorContractOpInfo &opInfo,
                       vector::ContractionOp contractOp);
  } // namespace GPU
} // namespace IREE
} // namespace iree_compiler
} // namespace mlir

#endif IREE_COMPILER_CODEGEN_LLVMGPU_INTERNALAPI_H_

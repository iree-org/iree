if(NOT "${IREE_TARGET_BACKEND_CUDA}")
  return()
endif()

iree_add_all_subdirs()

# TODO(#5893): iree_c_embed_data requires a relative path and we need it to be
# here to keep the path consistent with bazel build file. Fix it by making cmake
# target more flexible.
iree_c_embed_data(
  NAME
    cuda_libdevice
  SRCS
    "../../../../../../third_party/cuda/nvvm/libdevice/libdevice.10.bc"
  C_FILE_OUTPUT
    "libdevice.c"
  H_FILE_OUTPUT
    "libdevice.h"
  FLATTEN
  PUBLIC
)

iree_cc_library(
  NAME
    CUDA
  HDRS
    "CUDATarget.h"
  SRCS
    "CUDATarget.cpp"
  DEPS
    ::cuda_libdevice
    LLVMCore
    LLVMNVPTXCodeGen
    LLVMSupport
    LLVMTarget
    MLIRLLVMIR
    MLIRLLVMToLLVMIRTranslation
    MLIRNVVMIR
    MLIRNVVMToLLVMIRTranslation
    MLIRPass
    MLIRSupport
    MLIRTargetLLVMIRExport
    iree::base::internal::flatcc
    iree::compiler::Conversion::LinalgToNVVM
    iree::compiler::Dialect::HAL::Target
    iree::compiler::Utils
    iree::schemas::cuda_executable_def_c_fbs
  PUBLIC
)

### BAZEL_TO_CMAKE_PRESERVES_ALL_CONTENT_BELOW_THIS_LINE ###

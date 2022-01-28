# Copyright 2021 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# Doesn't use bazel_to_cmake because of various special logic throughout.

if(NOT "${IREE_TARGET_BACKEND_CUDA}")
  return()
endif()

iree_add_all_subdirs()

# iree_cc_library(
#   NAME
#     CUDA
#   HDRS
#     "CUDATarget.h"
#     "LLVMPasses.h"
#   SRCS
#     "CUDATarget.cpp"
#     "NoLoopUnrollPass.cpp"
#   DEPS
#     LLVMAnalysis
#     LLVMBitReader
#     LLVMCore
#     LLVMipo
#     LLVMLinker
#     LLVMNVPTXCodeGen
#     LLVMSupport
#     LLVMTarget
#     MLIRGPUOps
#     MLIRLLVMIR
#     MLIRLLVMToLLVMIRTranslation
#     MLIRNVVMIR
#     MLIRNVVMToLLVMIRTranslation
#     MLIRPass
#     MLIRSupport
#     MLIRTargetLLVMIRExport
#     iree::base::internal::flatcc::building
#     iree::compiler::Codegen::LLVMGPU
#     iree::compiler::Dialect::HAL::Target
#     iree::compiler::Utils
#     iree::schemas::cuda_executable_def_c_fbs
#   PUBLIC
# )

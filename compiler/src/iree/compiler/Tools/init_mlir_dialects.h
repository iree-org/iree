// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// This files defines a helper to trigger the registration of dialects to
// the system.
//
// Based on MLIR's InitAllDialects but without dialects we don't care about.

#ifndef IREE_COMPILER_TOOLS_INIT_MLIR_DIALECTS_H_
#define IREE_COMPILER_TOOLS_INIT_MLIR_DIALECTS_H_

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/TransformOps/AffineTransformOps.h"
#include "mlir/Dialect/ArmNeon/ArmNeonDialect.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Bufferization/TransformOps/BufferizationTransformOps.h"
#include "mlir/Dialect/Complex/IR/Complex.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/Extensions/InlinerExtension.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Func/TransformOps/FuncTransformOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/GPU/TransformOps/GPUTransformOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/TransformOps/DialectExtension.h"
#include "mlir/Dialect/MLProgram/IR/MLProgram.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/MemRef/TransformOps/MemRefTransformOps.h"
#include "mlir/Dialect/Mesh/IR/MeshOps.h"
#include "mlir/Dialect/NVGPU/IR/NVGPUDialect.h"
#include "mlir/Dialect/PDL/IR/PDL.h"
#include "mlir/Dialect/PDLInterp/IR/PDLInterp.h"
#include "mlir/Dialect/Quant/QuantOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/TransformOps/SCFTransformOps.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVDialect.h"
#include "mlir/Dialect/Shape/IR/Shape.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tensor/IR/TensorInferTypeOpInterfaceImpl.h"
#include "mlir/Dialect/Tensor/IR/TensorTilingInterfaceImpl.h"
#include "mlir/Dialect/Tensor/TransformOps/TensorTransformOps.h"
#include "mlir/Dialect/Transform/IR/TransformDialect.h"
#include "mlir/Dialect/Transform/LoopExtension/LoopExtension.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/Vector/TransformOps/VectorTransformOps.h"
#include "mlir/IR/Dialect.h"

#ifdef IREE_HAVE_C_OUTPUT_FORMAT
#include "mlir/Dialect/EmitC/IR/EmitC.h"
#endif // IREE_HAVE_C_OUTPUT_FORMAT

namespace mlir {

// Add all the MLIR dialects to the provided registry.
inline void registerMlirDialects(DialectRegistry &registry) {
  // clang-format off
  registry.insert<affine::AffineDialect,
                  bufferization::BufferizationDialect,
                  cf::ControlFlowDialect,
                  complex::ComplexDialect,
                  gpu::GPUDialect,
                  nvgpu::NVGPUDialect,
                  LLVM::LLVMDialect,
                  linalg::LinalgDialect,
                  math::MathDialect,
                  mesh::MeshDialect,
                  memref::MemRefDialect,
                  ml_program::MLProgramDialect,
                  pdl::PDLDialect,
                  pdl_interp::PDLInterpDialect,
                  scf::SCFDialect,
                  quant::QuantizationDialect,
                  spirv::SPIRVDialect,
                  arm_neon::ArmNeonDialect,
                  func::FuncDialect,
                  mlir::arith::ArithDialect,
                  vector::VectorDialect,
                  tensor::TensorDialect,
                  transform::TransformDialect,
                  shape::ShapeDialect>();
  // clang-format on
  func::registerInlinerExtension(registry);
  tensor::registerInferTypeOpInterfaceExternalModels(registry);
  tensor::registerTilingInterfaceExternalModels(registry);

  // Register all transform dialect extensions.
  affine::registerTransformDialectExtension(registry);
  bufferization::registerTransformDialectExtension(registry);
  func::registerTransformDialectExtension(registry);
  gpu::registerTransformDialectExtension(registry);
  linalg::registerTransformDialectExtension(registry);
  memref::registerTransformDialectExtension(registry);
  scf::registerTransformDialectExtension(registry);
  tensor::registerTransformDialectExtension(registry);
  transform::registerLoopExtension(registry);
  vector::registerTransformDialectExtension(registry);

#ifdef IREE_HAVE_C_OUTPUT_FORMAT
  registry.insert<emitc::EmitCDialect>();
#endif // IREE_HAVE_C_OUTPUT_FORMAT
}

} // namespace mlir

#endif // IREE_COMPILER_TOOLS_INIT_MLIR_DIALECTS_H_

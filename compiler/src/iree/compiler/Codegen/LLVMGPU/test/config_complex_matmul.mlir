// RUN: iree-opt --split-input-file --iree-gpu-test-target=gfx1201 \
// RUN:   --pass-pipeline='builtin.module(iree-llvmgpu-select-lowering-strategy)' %s \
// RUN:   | FileCheck %s
// RUN: iree-opt --split-input-file --iree-gpu-test-target=gfx1201 \
// RUN:   --pass-pipeline='builtin.module(linalg-generalize-named-ops,iree-llvmgpu-select-lowering-strategy)' %s \
// RUN:   | FileCheck %s

// Verify that complex element types do not crash the MMA heuristics and get
// routed to a working pipeline (#iree_gpu.pipeline<TileAndFuse> via the SIMT contraction
// config, not the MMA-based matmul config).

//      CHECK: #iree_codegen.translation_info<pipeline = #iree_gpu.pipeline<TileAndFuse>
// CHECK-LABEL: func.func @complex_batch_matmul
//      CHECK:   linalg.{{fill|generic}}
//      CHECK:   lowering_config = #iree_gpu.lowering_config
//  CHECK-NOT:   mma_kind

!TA = tensor<17x96x153xcomplex<f32>>
!TB = tensor<17x153x16xcomplex<f32>>
!TC = tensor<17x96x16xcomplex<f32>>
!DTC = !iree_tensor_ext.dispatch.tensor<writeonly:tensor<17x96x16xcomplex<f32>>>

func.func @complex_batch_matmul(%arg0: !TA, %arg1: !TB, %arg2: !DTC) {
  %zero = complex.constant [0.0 : f32, 0.0 : f32] : complex<f32>
  %init = tensor.empty() : tensor<17x96x16xcomplex<f32>>
  %filled = linalg.fill ins(%zero : complex<f32>) outs(%init : !TC) -> !TC
  %bmm = linalg.batch_matmul ins(%arg0, %arg1 : !TA, !TB) outs(%filled : !TC) -> !TC
  iree_tensor_ext.dispatch.tensor.store %bmm, %arg2, offsets = [0, 0, 0], sizes = [17, 96, 16], strides = [1, 1, 1] : !TC -> !DTC
  return
}

// -----

//      CHECK: #iree_codegen.translation_info<pipeline = #iree_gpu.pipeline<TileAndFuse>
// CHECK-LABEL: func.func @complex_matmul
//      CHECK:   linalg.{{fill|generic}}
//      CHECK:   lowering_config = #iree_gpu.lowering_config
//  CHECK-NOT:   mma_kind

!MA = tensor<64x128xcomplex<f32>>
!MB = tensor<128x32xcomplex<f32>>
!MC = tensor<64x32xcomplex<f32>>
!DMC = !iree_tensor_ext.dispatch.tensor<writeonly:tensor<64x32xcomplex<f32>>>

func.func @complex_matmul(%arg0: !MA, %arg1: !MB, %arg2: !DMC) {
  %zero = complex.constant [0.0 : f32, 0.0 : f32] : complex<f32>
  %init = tensor.empty() : tensor<64x32xcomplex<f32>>
  %filled = linalg.fill ins(%zero : complex<f32>) outs(%init : !MC) -> !MC
  %mm = linalg.matmul ins(%arg0, %arg1 : !MA, !MB) outs(%filled : !MC) -> !MC
  iree_tensor_ext.dispatch.tensor.store %mm, %arg2, offsets = [0, 0], sizes = [64, 32], strides = [1, 1] : !MC -> !DMC
  return
}

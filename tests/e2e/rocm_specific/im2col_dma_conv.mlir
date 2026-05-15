// Test conv2d using im2col + DMA path on gfx950+.
//
// Compile:
//   iree-compile \
//     --iree-hal-target-backends=rocm \
//     --iree-rocm-target=gfx950 \
//     --iree-codegen-llvmgpu-use-igemm=true \
//     --iree-llvmgpu-use-direct-load=true \
//     im2col_dma_conv.mlir -o im2col_dma_conv.vmfb
//
// Run:
//   iree-check-module --device=hip --module=im2col_dma_conv.vmfb
//
// Dump IR (for debugging):
//   iree-compile \
//     --iree-hal-target-backends=rocm \
//     --iree-rocm-target=gfx950 \
//     --iree-codegen-llvmgpu-use-igemm=true \
//     --iree-llvmgpu-use-direct-load=true \
//     --mlir-print-ir-after-all \
//     im2col_dma_conv.mlir -o im2col_dma_conv.vmfb 2> im2col_dma_ir_dump.mlir

!input_type = tensor<1x10x10x512xf16>
!filter_type = tensor<3x3x512x512xf16>
!output_type = tensor<1x8x8x512xf32>

func.func @im2col_dma_conv() {
  %input = util.unfoldable_constant dense<1.0> : !input_type
  %filter = util.unfoldable_constant dense<1.0> : !filter_type
  %cst = arith.constant 0.000000e+00 : f32
  %empty = tensor.empty() : !output_type
  %fill = linalg.fill ins(%cst : f32) outs(%empty : !output_type) -> !output_type
  %result = linalg.conv_2d_nhwc_hwcf {
    dilations = dense<1> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } ins(%input, %filter : !input_type, !filter_type)
    outs(%fill : !output_type) -> !output_type
  // Each output element = sum over 3*3*512 products of 1*1 = 4608.
  check.expect_almost_eq_const(
    %result, dense<4608.0> : !output_type) : !output_type
  return
}

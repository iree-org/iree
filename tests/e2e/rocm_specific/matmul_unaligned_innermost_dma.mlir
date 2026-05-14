// E2E test: fully-unaligned matmul (M, N, K all not multiples of the
// workgroup tile=64), exercising the coalesced_gather_dma straddle case
// that GPUPushDownDMABoundsToConsumers + the buffer_resource_cast
// validBytes hack address.
//
//   131x131x131 : 3x3=9 workgroups, 3 K-blocks. Tail K-block has straddle.
//                 Both LHS and RHS have non-DWORD-aligned innermost rows.
//
// All-1.0 inputs → output = K (= 131.0 here).
//
// Compile + run (assuming a build at /home/xunli/iree-build):
//
//   iree-compile \
//     --iree-hal-target-device=hip --iree-rocm-target=gfx950 \
//     --iree-llvmgpu-use-direct-load \
//     matmul_unaligned_innermost_dma.mlir \
//     -o /tmp/matmul_unaligned.vmfb
//
//   iree-run-module --device=hip \
//     --module=/tmp/matmul_unaligned.vmfb \
//     --function=matmul_f16_131x131x131 \
//     --input='131x131xf16=1.0' --input='131x131xf16=1.0' \
//     --expected_output='131x131xf32=131.0'

!acc131_t = tensor<131x131xf32>
func.func @matmul_f16_131x131x131(%lhs: tensor<131x131xf16>, %rhs: tensor<131x131xf16>) -> !acc131_t {
  %zero  = arith.constant 0.0 : f32
  %empty = tensor.empty() : !acc131_t
  %fill  = linalg.fill ins(%zero : f32) outs(%empty : !acc131_t) -> !acc131_t
  %res   = linalg.matmul ins(%lhs, %rhs : tensor<131x131xf16>, tensor<131x131xf16>)
                         outs(%fill : !acc131_t) -> !acc131_t
  return %res : !acc131_t
}

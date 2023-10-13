#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> (d0)>
func.func @matmul_reduction(%lhs : tensor<16x16xf16>, %rhs : tensor<16x16xf16>) -> tensor<16x16xf16> {
  %c0 = arith.constant 0.0 : f16
  %c1 = arith.constant -1.0e+04 : f16
  %acc = tensor.empty() : tensor<16xf16>
  %init = linalg.fill ins(%c1 : f16) outs(%acc : tensor<16xf16>) -> tensor<16xf16>
  %0 = tensor.empty() : tensor<16x16xf16>
  %1 = linalg.fill ins(%c0 : f16) outs(%0 : tensor<16x16xf16>) -> tensor<16x16xf16>
  %2 = linalg.matmul_transpose_b ins(%lhs, %rhs : tensor<16x16xf16>, tensor<16x16xf16>)
      outs(%1 : tensor<16x16xf16>) -> tensor<16x16xf16>
  %6 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "reduction"]}
        ins(%2 : tensor<16x16xf16>) outs(%init : tensor<16xf16>) {
        ^bb0(%in: f16, %out: f16):
          %20 = arith.maximumf %in, %out : f16
          linalg.yield %20 : f16
        } -> tensor<16xf16>
  %8 = linalg.generic {indexing_maps = [#map1, #map], iterator_types=["parallel", "parallel"]}
        ins(%6 : tensor<16xf16>) outs(%0 : tensor<16x16xf16>) {
        ^bb0(%in: f16,  %out: f16):
          linalg.yield %in : f16
        } -> tensor<16x16xf16>
  return %8 : tensor<16x16xf16>
}

// RUN: iree-compile %s --iree-hal-target-backends=cuda \
// RUN:     --iree-hal-cuda-llvm-target-arch=sm_80 \
// RUN:     --iree-codegen-llvmgpu-enable-transform-dialect-jit=false \
// RUN:     --iree-flow-dispatch-use-transform-dialect=%p/mma_reduction_layout_analysis_dispatch_spec.mlir \
// RUN:     --iree-codegen-llvmgpu-use-transform-dialect=%p/mma_reduction_layout_analysis_codegen_spec.mlir | \
// RUN: iree-run-module --module=- --function=matmul_reduction --device=cuda \
// RUN: --input="16x16xf16=[[3.0,2.0,2.5,4.5,1.5,4.0,2.0,2.5,4.0,4.0,1.5,0.5,2.0,3.0,0.5,2.0],[2.5,2.5,0.5,3.5,0.0,2.5,3.5,1.0,0.5,0.0,3.0,4.5,0.5,0.5,0.0,3.5],[4.5,3.0,4.0,2.5,1.0,0.5,0.0,4.5,0.0,2.5,3.5,0.0,2.0,4.5,1.5,4.5],[0.0,2.0,1.5,0.0,2.0,1.5,3.0,2.0,2.0,4.0,4.0,2.5,0.0,3.0,2.0,0.5],[0.5,3.5,3.0,2.5,0.0,2.5,3.0,3.0,4.5,2.0,2.0,1.0,2.0,1.0,3.5,2.0],[0.0,4.5,2.0,4.0,2.5,2.5,1.5,1.5,1.5,3.0,3.0,0.0,2.5,0.5,2.0,2.0],[3.5,4.0,3.5,1.5,2.0,0.5,1.0,2.5,4.0,3.5,0.0,3.0,0.0,1.5,4.5,0.0],[4.5,3.5,1.0,4.5,0.5,0.0,1.5,4.5,1.5,3.5,3.0,2.5,0.0,0.5,0.0,4.0],[2.0,3.0,0.5,2.0,1.5,0.5,2.0,2.5,2.5,4.0,2.0,4.5,4.0,0.0,2.0,3.0],[2.5,4.0,4.0,3.0,2.0,2.0,4.5,0.5,4.5,1.0,2.0,0.0,4.5,1.0,3.0,0.5],[4.0,1.5,3.5,3.0,2.5,4.5,1.0,3.5,3.0,2.5,2.5,2.0,2.0,4.5,1.5,2.5],[3.0,3.0,0.0,2.5,1.0,3.0,0.0,1.5,1.5,2.5,0.5,1.0,3.0,3.5,1.5,1.5],[0.0,4.5,0.5,1.5,0.5,4.0,3.5,4.0,4.0,0.0,0.5,1.0,4.5,1.5,0.0,3.5],[2.5,2.0,2.5,1.5,3.0,0.0,2.0,1.0,2.5,4.0,0.0,4.0,4.0,1.5,3.0,2.5],[3.0,0.0,4.0,4.0,2.0,0.5,1.0,3.5,4.0,2.5,4.0,4.5,0.0,3.0,1.5,2.5],[0.5,0.5,2.5,4.0,1.0,2.5,0.5,4.5,2.0,3.0,1.5,4.5,1.5,4.5,0.5,1.5]]" \
// RUN: --input="16x16xf16=[[3.5,3.0,4.5,3.0,3.0,0.0,2.0,2.5,2.0,0.0,4.5,2.5,0.5,0.0,4.0,3.5],[0.0,0.5,2.0,4.5,0.0,4.0,1.5,3.5,0.5,2.5,3.5,1.5,3.5,4.5,4.0,3.0],[3.0,3.5,2.5,1.5,1.5,1.5,0.5,4.5,0.0,3.5,4.0,0.0,0.0,2.0,0.5,1.0],[1.5,4.0,3.5,3.5,0.0,0.0,0.0,2.0,3.0,1.5,0.0,3.0,0.0,2.5,2.0,3.0],[3.5,4.0,2.5,1.5,3.0,2.0,3.0,4.5,1.5,3.0,2.0,3.5,2.5,4.5,0.5,3.5],[0.0,0.0,0.0,0.5,1.0,2.5,1.5,1.0,2.5,1.5,0.0,1.5,1.5,2.0,4.5,2.5],[4.0,1.5,3.0,2.5,2.5,3.5,2.0,4.0,1.5,2.5,0.5,4.0,1.0,4.5,3.5,0.0],[1.0,2.0,4.0,4.5,4.5,3.5,0.0,1.0,4.5,3.5,2.0,3.0,0.5,4.0,3.5,1.5],[1.0,0.0,2.5,4.5,0.0,2.0,0.0,2.5,3.0,4.0,2.5,0.5,3.5,0.0,3.5,1.0],[0.0,3.5,4.0,0.0,0.0,4.5,1.0,3.5,1.5,3.0,2.0,1.0,0.5,0.5,2.0,0.0],[1.5,0.0,4.5,2.0,4.5,4.5,3.5,3.0,2.5,4.5,0.5,0.5,0.0,4.5,0.0,4.0],[4.5,3.5,4.0,4.0,1.5,4.0,1.0,4.0,2.5,0.5,4.5,3.5,3.5,0.5,4.5,3.0],[0.0,3.0,2.5,1.0,1.5,2.0,1.0,1.5,4.0,2.5,3.5,1.0,3.5,2.5,3.5,4.5],[1.5,4.5,2.0,2.0,2.0,0.5,4.0,2.0,4.0,3.5,4.0,1.0,1.5,2.5,1.0,0.0],[0.0,0.0,1.0,2.5,3.5,2.5,4.0,0.0,2.0,2.0,4.5,0.5,1.0,3.5,3.0,2.5],[2.0,2.0,0.5,2.0,4.5,2.5,3.0,1.5,4.5,2.0,3.5,3.0,1.0,2.0,1.5,2.0]]" |\
// RUN: FileCheck %s --check-prefix=EXEC

//      EXEC: result[0]: hal.buffer_view
// EXEC-NEXT: 16x16xf16=[116 116 116 116 116 116 116 116 116 116 116 116 116 116 116 116][96.5 96.5 96.5 96.5 96.5 96.5 96.5 96.5 96.5 96.5 96.5 96.5 96.5 96.5 96.5 96.5][124.75 124.75 124.75 124.75 124.75 124.75 124.75 124.75 124.75 124.75 124.75 124.75 124.75 124.75 124.75 124.75][86.75 86.75 86.75 86.75 86.75 86.75 86.75 86.75 86.75 86.75 86.75 86.75 86.75 86.75 86.75 86.75][115.5 115.5 115.5 115.5 115.5 115.5 115.5 115.5 115.5 115.5 115.5 115.5 115.5 115.5 115.5 115.5][103.75 103.75 103.75 103.75 103.75 103.75 103.75 103.75 103.75 103.75 103.75 103.75 103.75 103.75 103.75 103.75][109 109 109 109 109 109 109 109 109 109 109 109 109 109 109 109][114.75 114.75 114.75 114.75 114.75 114.75 114.75 114.75 114.75 114.75 114.75 114.75 114.75 114.75 114.75 114.75][110.75 110.75 110.75 110.75 110.75 110.75 110.75 110.75 110.75 110.75 110.75 110.75 110.75 110.75 110.75 110.75][122.75 122.75 122.75 122.75 122.75 122.75 122.75 122.75 122.75 122.75 122.75 122.75 122.75 122.75 122.75 122.75][136.5 136.5 136.5 136.5 136.5 136.5 136.5 136.5 136.5 136.5 136.5 136.5 136.5 136.5 136.5 136.5][87.75 87.75 87.75 87.75 87.75 87.75 87.75 87.75 87.75 87.75 87.75 87.75 87.75 87.75 87.75 87.75][102.75 102.75 102.75 102.75 102.75 102.75 102.75 102.75 102.75 102.75 102.75 102.75 102.75 102.75 102.75 102.75][102.75 102.75 102.75 102.75 102.75 102.75 102.75 102.75 102.75 102.75 102.75 102.75 102.75 102.75 102.75 102.75][126.25 126.25 126.25 126.25 126.25 126.25 126.25 126.25 126.25 126.25 126.25 126.25 126.25 126.25 126.25 126.25][106 106 106 106 106 106 106 106 106 106 106 106 106 106 106 106]

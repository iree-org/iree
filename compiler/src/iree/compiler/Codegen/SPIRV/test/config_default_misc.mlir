// RUN: iree-opt --split-input-file --iree-gpu-test-target=vp_android_baseline_2022@vulkan --pass-pipeline='builtin.module(iree-spirv-select-lowering-strategy-pass)' %s | FileCheck %s

#map = affine_map<(d0, d1, d2) -> (d1)>
#map1 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
func.func @complex_view_as_real(%arg0: tensor<1x1x32x50x2xf32>, %arg1: tensor<50xcomplex<f32>>) -> tensor<32x50x2xf32> {
  %c1 = arith.constant 1 : index
  %c0 = arith.constant 0 : index
  %0 = tensor.empty() : tensor<32x50x2xf32>
  %1 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg1 : tensor<50xcomplex<f32>>) outs(%0 : tensor<32x50x2xf32>) {
  ^bb0(%in: complex<f32>, %out: f32):
    %2 = linalg.index 0 : index
    %3 = linalg.index 1 : index
    %extracted_0 = tensor.extract %arg0[%c0, %c0, %2, %3, %c0] : tensor<1x1x32x50x2xf32>
    %extracted_1 = tensor.extract %arg0[%c0, %c0, %2, %3, %c1] : tensor<1x1x32x50x2xf32>
    %4 = complex.create %extracted_0, %extracted_1 : complex<f32>
    %5 = complex.mul %4, %in : complex<f32>
    %6 = complex.re %5 : complex<f32>
    %7 = complex.im %5 : complex<f32>
    %8 = linalg.index 2 : index
    %9 = arith.cmpi eq, %8, %c0 : index
    %10 = arith.select %9, %6, %7 : f32
    linalg.yield %10 : f32
  } -> tensor<32x50x2xf32>
  return %1 : tensor<32x50x2xf32>
}

//  CHECK-DAG: #[[CONFIG:.+]] = #iree_codegen.lowering_config<tile_sizes = {{\[}}[16, 2, 2], [1, 1, 1]{{\]}}>
//  CHECK-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation_info<pipeline = SPIRVBaseDistribute workgroup_size = [2, 2, 16]>
//      CHECK: func.func @complex_view_as_real(
// CHECK-SAME:     translation_info = #[[TRANSLATION]]
//      CHECK:   linalg.generic
// CHECK-SAME:       lowering_config = #[[CONFIG]]

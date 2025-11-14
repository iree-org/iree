// RUN: iree-opt --split-input-file --iree-gpu-test-target=vp_android_baseline_2022@vulkan --pass-pipeline='builtin.module(iree-spirv-select-lowering-strategy-pass)' %s | FileCheck %s

#map = affine_map<(d0, d1, d2) -> (d1)>
#map1 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
func.func @complex_view_as_real(%4: tensor<1xi32>, %5: tensor<1x1x32x50x2xf32>, %9: tensor<50xcomplex<f32>>) -> tensor<32x50x2xf32> {
  %c1 = arith.constant 1 : index
  %c0 = arith.constant 0 : index
  %6 = tensor.empty() : tensor<32x50x2xf32>
  %extracted = tensor.extract %4[%c0] : tensor<1xi32>
  %7 = arith.extsi %extracted : i32 to i64
  %8 = arith.index_cast %7 : i64 to index
  %10 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%9 : tensor<50xcomplex<f32>>) outs(%6 : tensor<32x50x2xf32>) {
  ^bb0(%in: complex<f32>, %out: f32):
    %11 = linalg.index 0 : index
    %12 = linalg.index 1 : index
    %extracted_0 = tensor.extract %5[%c0, %c0, %11, %12, %c0] : tensor<1x1x32x50x2xf32>
    %extracted_1 = tensor.extract %5[%c0, %c0, %11, %12, %c1] : tensor<1x1x32x50x2xf32>
    %13 = complex.create %extracted_0, %extracted_1 : complex<f32>
    %14 = complex.mul %13, %in : complex<f32>
    %15 = complex.re %14 : complex<f32>
    %16 = complex.im %14 : complex<f32>
    %17 = linalg.index 2 : index
    %18 = arith.cmpi eq, %17, %c0 : index
    %19 = arith.select %18, %15, %16 : f32
    linalg.yield %19 : f32
  } -> tensor<32x50x2xf32>
  return %10 : tensor<32x50x2xf32>
}

//  CHECK-DAG: #[[CONFIG:.+]] = #iree_codegen.lowering_config<tile_sizes = {{\[}}[16, 2, 2], [1, 1, 1]{{\]}}>
//  CHECK-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation_info<pipeline = SPIRVBaseDistribute workgroup_size = [2, 2, 16]>
//      CHECK: func.func @complex_view_as_real(
// CHECK-SAME:     translation_info = #[[TRANSLATION]]
//      CHECK:   linalg.generic
// CHECK-SAME:       lowering_config = #[[CONFIG]]

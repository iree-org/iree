// RUN: iree-opt --split-input-file --iree-gpu-test-target=gfx908 --pass-pipeline='builtin.module(iree-llvmgpu-select-lowering-strategy)' %s | FileCheck %s

// gfx908 a.k.a. CDNA1 is used here as an example of a GPU target that we don't have ukernels for.
// No need to add many ukernels here, just a quick check that we correctly do not select a ukernel.

func.func @argmax_2d_f32i64(%arg0 : tensor<1x?xf32>) -> tensor<1xi64> attributes {
  hal.executable.target = #hal.executable.target<"rocm", "rocm-hsaco-fb", {ukernels = "all"}>
} {
  %c0_i64 = arith.constant 0 : i64
  %cst = arith.constant 0xFF800000 : f32
  %0 = tensor.empty() : tensor<1xi64>
  %1 = linalg.fill ins(%c0_i64 : i64) outs(%0 : tensor<1xi64>) -> tensor<1xi64>
  %2 = tensor.empty() : tensor<1xf32>
  %3 = linalg.fill ins(%cst : f32) outs(%2 : tensor<1xf32>) -> tensor<1xf32>
  %4:2 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>, affine_map<(d0, d1) -> (d0)>], iterator_types = ["parallel", "reduction"]} ins(%arg0 : tensor<1x?xf32>) outs(%3, %1 : tensor<1xf32>, tensor<1xi64>) {
  ^bb0(%in: f32, %out: f32, %out_0: i64):
    %5 = linalg.index 1 : index
    %6 = arith.index_cast %5 : index to i64
    %7 = arith.maximumf %in, %out : f32
    %8 = arith.cmpf ogt, %in, %out : f32
    %9 = arith.select %8, %6, %out_0 : i64
    linalg.yield %7, %9 : f32, i64
  } -> (tensor<1xf32>, tensor<1xi64>)
  return %4#1 : tensor<1xi64>
}

//   CHECK-NOT: lowering_config<{{.*}}ukernel
// CHECK-LABEL: func @argmax_2d_f32i64(
//       CHECK: linalg.generic
//   CHECK-NOT: hal.executable.objects

// RUN: iree-opt --split-input-file --pass-pipeline="builtin.module(func.func(iree-codegen-gpu-lower-to-ukernels,cse,canonicalize))" %s | FileCheck %s

func.func @argmax_f32i64_with_selected_ukernel(%arg0 : tensor<1x?xf32>) -> tensor<1xi64> attributes {
  hal.executable.target = #hal.executable.target<"rocm", "rocm-hsaco-fb", {ukernels = "all"}>
} {
  %c0_i64 = arith.constant 0 : i64
  %cst = arith.constant 0xFF800000 : f32
  %0 = tensor.empty() : tensor<1xi64>
  %1 = linalg.fill ins(%c0_i64 : i64) outs(%0 : tensor<1xi64>) -> tensor<1xi64>
  %2 = tensor.empty() : tensor<1xf32>
  %3 = linalg.fill ins(%cst : f32) outs(%2 : tensor<1xf32>) -> tensor<1xf32>
  %4:2 = linalg.generic {
        indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>, affine_map<(d0, d1) -> (d0)>],
        iterator_types = ["parallel", "reduction"]
      }
      ins(%arg0 : tensor<1x?xf32>) outs(%3, %1 : tensor<1xf32>, tensor<1xi64>)
      attrs = {
        // The hal.executable.objects are not essential to the lowering itself.
        // In practice, they could have been hoisted out already by the
        // HoistExecutableObjects pass. But in the important case of running
        // iree-codegen-gpu-lower-to-ukernels shortly after
        // iree-codegen-gpu-select-ukernels, they may not have been hoisted yet,
        // and in that case, we check that iree-codegen-gpu-lower-to-ukernels
        // correctly preserves them on the generated ukernel op.
        hal.executable.objects = [#hal.executable.object<{path = "some_ukernel_filename.bc", data = dense_resource<some_ukernel_filename.bc> : vector<6420xi8>}>],
        // The ukernel attribute is what is essential to the lowering.
        ukernel = #iree_codegen.ukernel_spec<name = "some_ukernel", def_attrs = {vm.import.module = "rocm"}>} {
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

//CHECK-LABEL: func @argmax_f32i64_with_selected_ukernel(
// CHECK-SAME:     %[[ARG0:[a-zA-Z0-9]+]]: tensor<1x?xf32>
//  CHECK-DAG:   %[[C1_index:.+]] = arith.constant 1 : index
//  CHECK-DAG:   %[[C0_i64:.+]] = arith.constant 0
//  CHECK-DAG:   %[[FILL:.+]] = linalg.fill ins(%[[C0_i64]]
//      CHECK:   %[[MICRO_KERNEL:.+]] = iree_codegen.ukernel.generic
//  CHECK-SAME:      {hal.executable.objects = [#hal.executable.object<{path = "some_ukernel_filename.bc", data = dense_resource<some_ukernel_filename.bc> : vector<6420xi8>}>]}
//  CHECK-SAME:      "some_ukernel"
// CHECK-SAME:       ins(%[[ARG0]] :
// CHECK-SAME:       outs(%[[FILL]] :
//      CHECK:   return %[[MICRO_KERNEL]]

// -----

func.func @argmax_f32i64_without_selected_ukernel(%arg0 : tensor<1x?xf32>) -> tensor<1xi64> attributes {
  hal.executable.target = #hal.executable.target<"rocm", "rocm-hsaco-fb", {ukernels = "all"}>
} {
  %c0_i64 = arith.constant 0 : i64
  %cst = arith.constant 0xFF800000 : f32
  %0 = tensor.empty() : tensor<1xi64>
  %1 = linalg.fill ins(%c0_i64 : i64) outs(%0 : tensor<1xi64>) -> tensor<1xi64>
  %2 = tensor.empty() : tensor<1xf32>
  %3 = linalg.fill ins(%cst : f32) outs(%2 : tensor<1xf32>) -> tensor<1xf32>
  %4:2 = linalg.generic {
        indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>, affine_map<(d0, d1) -> (d0)>],
        iterator_types = ["parallel", "reduction"]
      }
      ins(%arg0 : tensor<1x?xf32>) outs(%3, %1 : tensor<1xf32>, tensor<1xi64>)
      attrs = {
        hal.executable.objects = [#hal.executable.object<{path = "some_ukernel_filename.bc", data = dense_resource<some_ukernel_filename.bc> : vector<6420xi8>}>]} {
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

//CHECK-LABEL: func @argmax_f32i64_without_selected_ukernel(
//      CHECK-NOT: iree_codegen.ukernel.generic
//      CHECK: linalg.generic

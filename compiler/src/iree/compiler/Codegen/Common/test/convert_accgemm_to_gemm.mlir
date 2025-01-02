// RUN: iree-opt --split-input-file --iree-convert-accgemm-to-gemm %s | FileCheck %s

#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>

  func.func @accumulate_gemm() {
    %c0 = arith.constant 0 : index
    %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) : !flow.dispatch.tensor<readonly:tensor<512x128xi8>>
    %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0) : !flow.dispatch.tensor<readonly:tensor<512x128xi8>>
    %2 = hal.interface.binding.subspan layout(#pipeline_layout) binding(2) alignment(64) offset(%c0) : !flow.dispatch.tensor<readwrite:tensor<512x512xi32>>
    %3 = flow.dispatch.tensor.load %0, offsets = [0, 0], sizes = [512, 128], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<512x128xi8>> -> tensor<512x128xi8>
    %4 = flow.dispatch.tensor.load %1, offsets = [0, 0], sizes = [512, 128], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<512x128xi8>> -> tensor<512x128xi8>
    %5 = flow.dispatch.tensor.load %2, offsets = [0, 0], sizes = [512, 512], strides = [1, 1] : !flow.dispatch.tensor<readwrite:tensor<512x512xi32>> -> tensor<512x512xi32>
    %6 = linalg.generic {
      indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>,
                       affine_map<(d0, d1, d2) -> (d1, d2)>,
                       affine_map<(d0, d1, d2) -> (d0, d1)>],
      iterator_types = ["parallel", "parallel", "reduction"]}
      ins(%3, %4 : tensor<512x128xi8>, tensor<512x128xi8>) outs(%5 : tensor<512x512xi32>) {
        ^bb0(%in: i8, %in_0: i8, %out: i32):
          %7 = arith.extsi %in : i8 to i32
          %8 = arith.extsi %in_0 : i8 to i32
          %9 = arith.muli %7, %8 : i32
          %10 = arith.addi %out, %9 : i32
          linalg.yield %10 : i32
       } -> tensor<512x512xi32>
    flow.dispatch.tensor.store %6, %2, offsets = [0, 0], sizes = [512, 512], strides = [1, 1] : tensor<512x512xi32> -> !flow.dispatch.tensor<readwrite:tensor<512x512xi32>>
    return
  }

// CHECK-LABEL: func.func @accumulate_gemm()
//       CHECK: %[[C0:.+]] = arith.constant 0 : i32
//       CHECK: %[[EMPTY:.+]] = tensor.empty() : tensor<512x512xi32>
//       CHECK: %[[FILL:.+]] = linalg.fill ins(%[[C0]] : i32) outs(%[[EMPTY]] : tensor<512x512xi32>) -> tensor<512x512xi32>
//       CHECK: %[[GEMM:.+]] = linalg.generic {{.*}} outs(%[[FILL]] : tensor<512x512xi32>) {
//       CHECK: %[[ADD:.+]] = linalg.generic {{.+}} ins(%[[GEMM]]
//       CHECK: flow.dispatch.tensor.store %[[ADD]]

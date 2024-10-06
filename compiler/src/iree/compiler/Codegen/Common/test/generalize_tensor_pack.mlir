// RUN: iree-opt %s --split-input-file --iree-codegen-generalize-tensor-pack | FileCheck %s

func.func @not_transpose() {
  %c0 = arith.constant 0 : index
  %c24576 = arith.constant 24576 : index
  %0 = hal.interface.binding.subspan layout(<bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>) binding(0) alignment(64) offset(%c0) flags("ReadOnly|Indirect") : !flow.dispatch.tensor<readonly:tensor<768x768xf32>>
  %1 = hal.interface.binding.subspan layout(<bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>) binding(1) alignment(64) offset(%c24576) flags(Indirect) : !flow.dispatch.tensor<writeonly:tensor<192x768x4x1xf32>>
  %2 = flow.dispatch.tensor.load %0, offsets = [0, 0], sizes = [768, 768], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<768x768xf32>> -> tensor<768x768xf32>
  %3 = tensor.empty() : tensor<192x768x4x1xf32>
  %pack = tensor.pack %2 outer_dims_perm = [0, 1] inner_dims_pos = [0, 1] inner_tiles = [4, 1] into %3 : tensor<768x768xf32> -> tensor<192x768x4x1xf32>
  flow.dispatch.tensor.store %pack, %1, offsets = [0, 0, 0, 0], sizes = [192, 768, 4, 1], strides = [1, 1, 1, 1] : tensor<192x768x4x1xf32> -> !flow.dispatch.tensor<writeonly:tensor<192x768x4x1xf32>>
  return
}
// CHECK-LABEL:   func.func @not_transpose
// CHECK:         %[[PACK:.*]] =  tensor.pack {{.*}} tensor<768x768xf32> -> tensor<192x768x4x1xf32>
// CHECK:          flow.dispatch.tensor.store %[[PACK]]

// -----

func.func @tranpose() {
  %c0 = arith.constant 0 : index
  %0 = hal.interface.binding.subspan layout(<bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>) binding(0) alignment(64) offset(%c0) flags("ReadOnly|Indirect") : !flow.dispatch.tensor<readonly:tensor<8x768xf32>>
  %1 = hal.interface.binding.subspan layout(<bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>) binding(1) alignment(64) offset(%c0) flags(Indirect) : !flow.dispatch.tensor<writeonly:tensor<1x768x8x1xf32>>
  %2 = flow.dispatch.tensor.load %0, offsets = [0, 0], sizes = [8, 768], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<8x768xf32>> -> tensor<8x768xf32>
  %3 = tensor.empty() : tensor<1x768x8x1xf32>
  %pack = tensor.pack %2 outer_dims_perm = [0, 1] inner_dims_pos = [0, 1] inner_tiles = [8, 1] into %3 : tensor<8x768xf32> -> tensor<1x768x8x1xf32>
  flow.dispatch.tensor.store %pack, %1, offsets = [0, 0, 0, 0], sizes = [1, 768, 8, 1], strides = [1, 1, 1, 1] : tensor<1x768x8x1xf32> -> !flow.dispatch.tensor<writeonly:tensor<1x768x8x1xf32>>
  return
}

// CHECK-LABEL:   func.func @tranpose
// CHECK:         %[[TRANSPOSE:.*]] =  linalg.transpose ins(%{{.*}} : tensor<8x768xf32>) outs(%{{.*}} : tensor<768x8xf32>) permutation = [1, 0]
// CHECK:          flow.dispatch.tensor.store %[[TRANSPOSE]]

// -----

func.func @not_store_consumed() {
  %c0 = arith.constant 0 : index
  %0 = hal.interface.binding.subspan layout(<bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>) binding(0) alignment(64) offset(%c0) flags("ReadOnly|Indirect") : !flow.dispatch.tensor<readonly:tensor<8x768xf32>>
  %1 = hal.interface.binding.subspan layout(<bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>) binding(1) alignment(64) offset(%c0) flags(Indirect) : !flow.dispatch.tensor<writeonly:tensor<1x768x8x1xf32>>
  %2 = flow.dispatch.tensor.load %0, offsets = [0, 0], sizes = [8, 768], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<8x768xf32>> -> tensor<8x768xf32>
  %3 = tensor.empty() : tensor<1x768x8x1xf32>
  %pack = tensor.pack %2 outer_dims_perm = [0, 1] inner_dims_pos = [0, 1] inner_tiles = [8, 1] into %3 : tensor<8x768xf32> -> tensor<1x768x8x1xf32>
  %arith = arith.negf %pack : tensor<1x768x8x1xf32>
  flow.dispatch.tensor.store %arith, %1, offsets = [0, 0, 0, 0], sizes = [1, 768, 8, 1], strides = [1, 1, 1, 1] : tensor<1x768x8x1xf32> -> !flow.dispatch.tensor<writeonly:tensor<1x768x8x1xf32>>
  return
}

// CHECK-LABEL:   func.func @not_store_consumed
// CHECK:         %[[PACK:.*]] = tensor.pack {{.*}} tensor<8x768xf32> -> tensor<1x768x8x1xf32>
// CHECK:         %[[ARITH:.*]] = arith.negf %[[PACK]] : tensor<1x768x8x1xf32>
// CHECK:          flow.dispatch.tensor.store %[[ARITH]]

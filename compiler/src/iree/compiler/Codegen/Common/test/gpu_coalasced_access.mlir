// RUN: iree-opt --iree-gpu-coalasced-access --split-input-file %s | FileCheck %s

func.func @dispatch_region() {
  %c0 = arith.constant 0 : index
  %c536870912 = arith.constant 536870912 : index
  %c1073741824 = arith.constant 1073741824 : index
  %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) offset(%c0) alignment(64) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<512x512x512xf32>>
  %1 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) offset(%c536870912) alignment(64) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<512x512x512xf32>>
  %2 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) offset(%c1073741824) alignment(64) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<512x512x512xf32>>
  %3 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) offset(%c0) alignment(64) flags(WriteOnly) : !flow.dispatch.tensor<writeonly:tensor<512x512x512xf32>>
  %4 = flow.dispatch.tensor.load %0, offsets = [0, 0, 0], sizes = [512, 512, 512], strides = [1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<512x512x512xf32>> -> tensor<512x512x512xf32>
  %5 = flow.dispatch.tensor.load %1, offsets = [0, 0, 0], sizes = [512, 512, 512], strides = [1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<512x512x512xf32>> -> tensor<512x512x512xf32>
  %6 = flow.dispatch.tensor.load %2, offsets = [0, 0, 0], sizes = [512, 512, 512], strides = [1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<512x512x512xf32>> -> tensor<512x512x512xf32>
  %7 = tensor.empty() : tensor<512x512x512xf32>
  %8 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d1, d0, d2)>, affine_map<(d0, d1, d2) -> (d1, d2, d1)>, affine_map<(d0, d1, d2) -> (d1, d2, d0)>, affine_map<(d0, d1, d2) -> (d1, d2, d0)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%4, %5, %6 : tensor<512x512x512xf32>, tensor<512x512x512xf32>, tensor<512x512x512xf32>) outs(%7 : tensor<512x512x512xf32>) {
  ^bb0(%in: f32, %in_0: f32, %in_1: f32, %out: f32):
    %9 = arith.divf %in, %in_1 : f32
    %10 = arith.mulf %9, %in_0 : f32
    linalg.yield %10 : f32
  } -> tensor<512x512x512xf32>
  flow.dispatch.tensor.store %8, %3, offsets = [0, 0, 0], sizes = [512, 512, 512], strides = [1, 1, 1] : tensor<512x512x512xf32> -> !flow.dispatch.tensor<writeonly:tensor<512x512x512xf32>>
  return
}

//  CHECK: func.func @dispatch_region()
//  CHECK:   %[[C0:.+]] = arith.constant 0 : index
//  CHECK:   %[[c536870912:.+]] = arith.constant 536870912 : index
//  CHECK:   %[[c1073741824:.+]] = arith.constant 1073741824 : index
//  CHECK:   hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) offset(%[[C0]]) alignment(64) flags("ReadOnly|Streaming")
//  CHECK:   hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) offset(%[[c536870912]]) alignment(64) flags(ReadOnly)
//  CHECK:   hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) offset(%[[c1073741824]]) alignment(64) flags(ReadOnly)
//  CHECK:   hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) offset(%[[C0]]) alignment(64) flags(WriteOnly)

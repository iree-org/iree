// RUN: iree-opt %s -allow-unregistered-dialect --split-input-file -iree-codegen-gpu-tensor-alloc | FileCheck %s

func.func @matmul_2048x512x1024() {
  %c0 = arith.constant 0 : index
  %cst = arith.constant 0.000000e+00 : f32
  %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<readonly:tensor<2048x1024xf32>>
  %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<readonly:tensor<1024x512xf32>>
  %2 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<writeonly:tensor<2048x512xf32>>
  %workgroup_id_x = hal.interface.workgroup.id[0] : index
  %workgroup_id_y = hal.interface.workgroup.id[1] : index
  %3 = affine.apply affine_map<()[s0] -> (s0 * 32)>()[%workgroup_id_y]
  %4 = affine.apply affine_map<()[s0] -> (s0 * 128)>()[%workgroup_id_x]
  %5 = flow.dispatch.tensor.load %2, offsets = [%3, %4], sizes = [32, 128], strides = [1, 1] : !flow.dispatch.tensor<writeonly:tensor<2048x512xf32>> -> tensor<32x128xf32>
  %6 = flow.dispatch.tensor.load %0, offsets = [%3, 0], sizes = [32, 1024], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<2048x1024xf32>> -> tensor<32x1024xf32>
  %7 = flow.dispatch.tensor.load %1, offsets = [0, %4], sizes = [1024, 128], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<1024x512xf32>> -> tensor<1024x128xf32>
  %8 = linalg.fill {lowering_config = #iree_codegen.lowering_config<tile_sizes = [[32, 128, 32]]>} ins(%cst : f32) outs(%5 : tensor<32x128xf32>) -> tensor<32x128xf32>
  %9 = linalg.matmul {lowering_config = #iree_codegen.lowering_config<tile_sizes = [[32, 128, 32]]>} ins(%6, %7 : tensor<32x1024xf32>, tensor<1024x128xf32>) outs(%8 : tensor<32x128xf32>) -> tensor<32x128xf32>
  flow.dispatch.tensor.store %9, %2, offsets = [%3, %4], sizes = [32, 128], strides = [1, 1] : tensor<32x128xf32> -> !flow.dispatch.tensor<writeonly:tensor<2048x512xf32>>
  return
}

//    CHECK-LABEL: func.func @matmul_2048x512x1024
//         CHECK:    scf.for {{.*}} -> (tensor<32x128xf32>) {
//         CHECK:      %[[A:.*]] = tensor.extract_slice %{{.*}}[0, %{{.*}}] [32, 32] [1, 1] : tensor<32x1024xf32> to tensor<32x32xf32>
//         CHECK:      %[[B:.*]] = tensor.extract_slice %{{.*}}[%{{.*}}, 0] [32, 128] [1, 1] : tensor<1024x128xf32> to tensor<32x128xf32>
//         CHECK:      %[[PA:.*]] = bufferization.alloc_tensor() copy(%[[A]]) {bufferization.escape = [false]} : tensor<32x32xf32>
//         CHECK:      %[[PB:.*]] = bufferization.alloc_tensor() copy(%[[B]]) {bufferization.escape = [false]} : tensor<32x128xf32>
//         CHECK:      %[[M:.*]] = linalg.matmul {{.*}} ins(%[[PA]], %[[PB]] : tensor<32x32xf32>, tensor<32x128xf32>) outs(%{{.*}} : tensor<32x128xf32>) -> tensor<32x128xf32>
//         CHECK:      scf.yield %[[M]] : tensor<32x128xf32>

// -----

func.func @matmul_1x384x384() {
  %c0 = arith.constant 0 : index
  %cst = arith.constant 0.000000e+00 : f32
  %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<readonly:tensor<1x384xf32>>
  %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<readonly:tensor<384x384xf32>>
  %2 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<writeonly:tensor<1x384xf32>>
  %3 = flow.dispatch.tensor.load %0, offsets = [0, 0], sizes = [1, 384], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<1x384xf32>> -> tensor<1x384xf32>
  %workgroup_id_x = hal.interface.workgroup.id[0] : index
  %4 = affine.apply affine_map<()[s0] -> (s0 * 128)>()[%workgroup_id_x]
  %5 = flow.dispatch.tensor.load %2, offsets = [0, %4], sizes = [1, 128], strides = [1, 1] : !flow.dispatch.tensor<writeonly:tensor<1x384xf32>> -> tensor<1x128xf32>
  %6 = flow.dispatch.tensor.load %1, offsets = [0, %4], sizes = [384, 128], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<384x384xf32>> -> tensor<384x128xf32>
  %7 = linalg.fill {lowering_config = #iree_codegen.lowering_config<tile_sizes = [[0, 128, 8]]>} ins(%cst : f32) outs(%5 : tensor<1x128xf32>) -> tensor<1x128xf32>
  %8 = linalg.matmul {lowering_config = #iree_codegen.lowering_config<tile_sizes = [[0, 128, 8]]>} ins(%3, %6 : tensor<1x384xf32>, tensor<384x128xf32>) outs(%7 : tensor<1x128xf32>) -> tensor<1x128xf32>
  flow.dispatch.tensor.store %8, %2, offsets = [0, %4], sizes = [1, 128], strides = [1, 1] : tensor<1x128xf32> -> !flow.dispatch.tensor<writeonly:tensor<1x384xf32>>
  return
}

//    CHECK-LABEL: func.func @matmul_1x384x384
//      CHECK-NOT:   bufferization.alloc_tensor()
//          CHECK: return

// -----

func.func @matmul_multi_uses() {
  %c0 = arith.constant 0 : index
  %cst = arith.constant 0.000000e+00 : f32
  %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<readonly:tensor<2048x1024xf32>>
  %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<readonly:tensor<1024x512xf32>>
  %2 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<writeonly:tensor<2048x512xf32>>
  %workgroup_id_x = hal.interface.workgroup.id[0] : index
  %workgroup_id_y = hal.interface.workgroup.id[1] : index
  %3 = affine.apply affine_map<()[s0] -> (s0 * 32)>()[%workgroup_id_y]
  %4 = affine.apply affine_map<()[s0] -> (s0 * 128)>()[%workgroup_id_x]
  %5 = flow.dispatch.tensor.load %2, offsets = [%3, %4], sizes = [32, 128], strides = [1, 1] : !flow.dispatch.tensor<writeonly:tensor<2048x512xf32>> -> tensor<32x128xf32>
  %6 = flow.dispatch.tensor.load %0, offsets = [%3, 0], sizes = [32, 1024], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<2048x1024xf32>> -> tensor<32x1024xf32>
  %7 = flow.dispatch.tensor.load %1, offsets = [0, %4], sizes = [1024, 128], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<1024x512xf32>> -> tensor<1024x128xf32>
  %8 = linalg.fill ins(%cst : f32) outs(%5 : tensor<32x128xf32>) -> tensor<32x128xf32>
  %9 = linalg.matmul ins(%6, %7 : tensor<32x1024xf32>, tensor<1024x128xf32>) outs(%8 : tensor<32x128xf32>) -> tensor<32x128xf32>
  "some_use"(%6) : (tensor<32x1024xf32>) -> ()
  flow.dispatch.tensor.store %9, %2, offsets = [%3, %4], sizes = [32, 128], strides = [1, 1] : tensor<32x128xf32> -> !flow.dispatch.tensor<writeonly:tensor<2048x512xf32>>
  return
}

// test corner case where the promoted value has multiple uses.
//    CHECK-LABEL: func.func @matmul_multi_uses
//         CHECK:    %[[C:.*]] = flow.dispatch.tensor.load
//         CHECK:    %[[A:.*]] = flow.dispatch.tensor.load
//         CHECK:    %[[B:.*]] = flow.dispatch.tensor.load
//         CHECK:    %[[PA:.*]] = bufferization.alloc_tensor() copy(%[[A]]) {bufferization.escape = [false]} : tensor<32x1024xf32>
//         CHECK:    %[[PB:.*]] = bufferization.alloc_tensor() copy(%[[B]]) {bufferization.escape = [false]} : tensor<1024x128xf32>
//         CHECK:    %[[M:.*]] = linalg.matmul {{.*}} ins(%[[PA]], %[[PB]] : tensor<32x1024xf32>, tensor<1024x128xf32>) outs(%{{.*}} : tensor<32x128xf32>) -> tensor<32x128xf32>
//         CHECK:    "some_use"(%[[A]]) : (tensor<32x1024xf32>) -> ()

// -----

  func.func @matmul_33x33x903168_f32() {
    %c0 = arith.constant 0 : index
    %cst = arith.constant 0.000000e+00 : f32
    %cst_0 = arith.constant 9.031680e+05 : f32
    %cst_1 = arith.constant 0.949999988 : f32
    %c32 = arith.constant 32 : index
    %workgroup_id_x = hal.interface.workgroup.id[0] : index
    %0 = affine.min affine_map<()[s0] -> (s0 * -32 + 33, 32)>()[%workgroup_id_x]
    %1 = arith.cmpi eq, %0, %c32 : index
    scf.if %1 {
      %2 = hal.interface.constant.load[0] : i32
      %3 = hal.interface.constant.load[1] : i32
      %4 = hal.interface.constant.load[2] : i32
      %5 = arith.index_castui %2 {stream.alignment = 4096 : index, stream.values = [1240289280 : index, 1789415424 : index]} : i32 to index
      %6 = arith.index_castui %3 {stream.alignment = 8192 : index, stream.values = [633077760 : index, 752295936 : index]} : i32 to index
      %7 = arith.index_castui %4 {stream.alignment = 64 : index, stream.values = [1486349952 : index, 1486358464 : index]} : i32 to index
      %8 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%5) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<33x903168xf32>>
      %9 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%6) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<903168x33xf32>>
      %10 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<33x33xf32>>
      %11 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) alignment(64) offset(%7) : !flow.dispatch.tensor<writeonly:tensor<33x33xf32>>
      %12 = affine.apply affine_map<()[s0] -> (s0 * 32)>()[%workgroup_id_x]
      %13 = flow.dispatch.tensor.load %11, offsets = [%12, 0], sizes = [32, 33], strides = [1, 1] : !flow.dispatch.tensor<writeonly:tensor<33x33xf32>> -> tensor<32x33xf32>
      %14 = flow.dispatch.tensor.load %9, offsets = [0, 0], sizes = [903168, 33], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<903168x33xf32>> -> tensor<903168x33xf32>
      %15 = flow.dispatch.tensor.load %10, offsets = [%12, 0], sizes = [32, 33], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<33x33xf32>> -> tensor<32x33xf32>
      %16 = flow.dispatch.tensor.load %8, offsets = [%12, 0], sizes = [32, 903168], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<33x903168xf32>> -> tensor<32x903168xf32>
      %17 = linalg.fill {lowering_config = #iree_codegen.lowering_config<tile_sizes = [[32, 128, 32]]>} ins(%cst : f32) outs(%13 : tensor<32x33xf32>) -> tensor<32x33xf32>
      %18 = linalg.matmul {lowering_config = #iree_codegen.lowering_config<tile_sizes = [[32, 128, 32]]>} ins(%16, %14 : tensor<32x903168xf32>, tensor<903168x33xf32>) outs(%17 : tensor<32x33xf32>) -> tensor<32x33xf32>
      %19 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%15 : tensor<32x33xf32>) outs(%18 : tensor<32x33xf32>) attrs =  {lowering_config = #iree_codegen.lowering_config<tile_sizes = [[32, 128, 32]]>} {
      ^bb0(%in: f32, %out: f32):
        %20 = arith.divf %out, %cst_0 : f32
        %21 = arith.mulf %in, %cst_1 : f32
        %22 = arith.addf %21, %20 : f32
        linalg.yield %22 : f32
      } -> tensor<32x33xf32>
      flow.dispatch.tensor.store %19, %11, offsets = [%12, 0], sizes = [32, 33], strides = [1, 1] : tensor<32x33xf32> -> !flow.dispatch.tensor<writeonly:tensor<33x33xf32>>
    }
    return
  }

// The allocation should not happen when there is any unaligned size, e.g., 33 in this case.
//
// CHECK-LABEL: func.func @matmul_33x33x903168_f32
// CHECK-NOT: bufferization.alloc_tensor()


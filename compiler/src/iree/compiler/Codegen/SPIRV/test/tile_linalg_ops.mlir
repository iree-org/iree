// RUN: iree-opt -split-input-file --pass-pipeline="builtin.module(func.func(iree-spirv-tile))" %s | FileCheck %s

func.func @innermost_reduction() {
  %c1 = arith.constant 1 : index
  %c128 = arith.constant 128 : index
  %cst = arith.constant -0.000000e+00 : f32
  %0 = hal.interface.constant.load[0] : i32
  %1 = hal.interface.constant.load[1] : i32
  %2 = hal.interface.constant.load[2] : i32
  %3 = arith.index_cast %0 {stream.alignment = 512 : index, stream.values = [0 : index, 394752 : index, 984064 : index]} : i32 to index
  %4 = arith.index_cast %1 {stream.alignment = 512 : index, stream.values = [0 : index, 196608 : index, 197120 : index]} : i32 to index
  %5 = arith.index_cast %2 {stream.alignment = 512 : index, stream.values = [512 : index, 197120 : index, 197632 : index]} : i32 to index
  %6 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%3) : !flow.dispatch.tensor<readonly:tensor<128x384xf32>>
  %7 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%4) : !flow.dispatch.tensor<readonly:tensor<128xf32>>
  %8 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%5) : !flow.dispatch.tensor<writeonly:tensor<128xf32>>
  %workgroup_id_x = hal.interface.workgroup.id[0] : index
  %workgroup_count_x = hal.interface.workgroup.count[0] : index
  %9 = affine.apply affine_map<()[s0] -> (s0 * 128)>()[%workgroup_id_x]
  %10 = affine.apply affine_map<()[s0] -> (s0 * 128)>()[%workgroup_count_x]
  scf.for %arg0 = %9 to %c128 step %10 {
    %11 = flow.dispatch.tensor.load %6, offsets = [%arg0, 0], sizes = [128, 384], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<128x384xf32>> -> tensor<128x384xf32>
    %12 = flow.dispatch.tensor.load %7, offsets = [%arg0], sizes = [128], strides = [1] : !flow.dispatch.tensor<readonly:tensor<128xf32>> -> tensor<128xf32>
    %13 = tensor.empty() : tensor<128xf32>
    %14 = linalg.fill ins(%cst : f32) outs(%13 : tensor<128xf32>) -> tensor<128xf32>
    %15 = linalg.generic {
      indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>, affine_map<(d0, d1) -> (d0)>],
      iterator_types = ["parallel", "reduction"]
    } ins(%11, %12 : tensor<128x384xf32>, tensor<128xf32>) outs(%14 : tensor<128xf32>)
    attrs = {lowering_config = #iree_codegen.lowering_config<tile_sizes = [[128], [4], [0, 4]]>} {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %16 = arith.subf %arg1, %arg2 : f32
      %17 = arith.mulf %16, %16 : f32
      %18 = arith.addf %17, %arg3 : f32
      linalg.yield %18 : f32
    } -> tensor<128xf32>
    flow.dispatch.tensor.store %15, %8, offsets = [%arg0], sizes = [128], strides = [%c1] : tensor<128xf32> -> !flow.dispatch.tensor<writeonly:tensor<128xf32>>
  }
  return
}

// CHECK-LABEL: func @innermost_reduction()

//  CHECK-DAG:  %[[C0:.+]] = arith.constant 0 : index
//  CHECK-DAG:  %[[C4:.+]] = arith.constant 4 : index
//  CHECK-DAG:  %[[C128:.+]] = arith.constant 128 : index
//  CHECK-DAG:  %[[C384:.+]] = arith.constant 384 : index

//      CHECK: scf.for
//      CHECK:   scf.for %{{.+}} = %[[C0]] to %[[C128]] step %[[C4]]
//      CHECK:     linalg.fill
//      CHECK:     scf.for %{{.+}} = %[[C0]] to %[[C384]] step %[[C4]]
//      CHECK:       linalg.generic
// CHECK-SAME:         ins(%{{.+}}, %{{.+}} : tensor<4x4xf32>, tensor<4xf32>)
// CHECK-SAME:         outs(%{{.+}}g4 : tensor<4xf32>)

// -----

func.func @has_scf_if() {
  %c49152 = arith.constant 49152 : index
  %c0 = arith.constant 0 : index
  %c4096_i32 = arith.constant 4096 : i32
  %c1023_i32 = arith.constant 1023 : i32
  %c2_i32 = arith.constant 2 : i32
  %c0_i32 = arith.constant 0 : i32
  %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<49152xi32>>
  %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<readwrite:tensor<49152xi32>>
  %workgroup_id_x = hal.interface.workgroup.id[0] : index
  %workgroup_count_x = hal.interface.workgroup.count[0] : index
  %2 = affine.apply affine_map<()[s0] -> (s0 * 256)>()[%workgroup_id_x]
  %3 = affine.apply affine_map<()[s0] -> (s0 * 256)>()[%workgroup_count_x]
  scf.for %arg0 = %2 to %c49152 step %3 {
    %4 = flow.dispatch.tensor.load %0, offsets = [%arg0], sizes = [256], strides = [1] : !flow.dispatch.tensor<readonly:tensor<49152xi32>> -> tensor<256xi32>
    %5 = flow.dispatch.tensor.load %1, offsets = [%arg0], sizes = [256], strides = [1] : !flow.dispatch.tensor<readwrite:tensor<49152xi32>> -> tensor<256xi32>
    %6 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%4 : tensor<256xi32>) outs(%5 : tensor<256xi32>) attrs =  {lowering_config = #iree_codegen.lowering_config<tile_sizes = [[256], [4]]>} {
    ^bb0(%in: i32, %out: i32):
      %7 = arith.cmpi sle, %in, %c0_i32 : i32
      %8 = scf.if %7 -> (i32) {
        scf.yield %c0_i32 : i32
      } else {
        %9 = arith.cmpi sge, %in, %c4096_i32 : i32
        %10 = scf.if %9 -> (i32) {
          scf.yield %c1023_i32 : i32
        } else {
          %11 = arith.shrsi %in, %c2_i32 : i32
          scf.yield %11 : i32
        }
        scf.yield %10 : i32
      }
      linalg.yield %8 : i32
    } -> tensor<256xi32>
    flow.dispatch.tensor.store %6, %1, offsets = [%arg0], sizes = [256], strides = [1] : tensor<256xi32> -> !flow.dispatch.tensor<readwrite:tensor<49152xi32>>
  }
  return
}

// CHECK-LABEL: func @has_scf_if()

//  CHECK-DAG:  %[[C0:.+]] = arith.constant 0 : index
//  CHECK-DAG:  %[[C4:.+]] = arith.constant 4 : index
//  CHECK-DAG:  %[[C256:.+]] = arith.constant 256 : index

//      CHECK: scf.for
//      CHECK:     scf.for %{{.+}} = %[[C0]] to %[[C256]] step %[[C4]]
//      CHECK:       linalg.generic
// CHECK-SAME:         ins(%{{.+}}slice : tensor<4xi32>)
// CHECK-SAME:         outs(%{{.+}} : tensor<4xi32>)
//      CHECK: scf.if
//      CHECK: scf.if
//      CHECK: linalg.yield %{{.*}} : i32

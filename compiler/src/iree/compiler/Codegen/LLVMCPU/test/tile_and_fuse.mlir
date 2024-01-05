// RUN: iree-opt --pass-pipeline="builtin.module(func.func(iree-llvmcpu-tile-and-fuse{tiling-level=0}))" --split-input-file %s | FileCheck %s

func.func @matmul_bias_add(%arg0 : tensor<?x?xf32>, %arg1 : tensor<?x?xf32>, %arg2 : tensor<?xf32>) -> tensor<?x?xf32> {
  %cst = arith.constant 0.0 : f32
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %d0 = tensor.dim %arg0, %c0 : tensor<?x?xf32>
  %d1 = tensor.dim %arg1, %c1 : tensor<?x?xf32>
  %init = tensor.empty(%d0, %d1) : tensor<?x?xf32>
  %0 = linalg.fill ins(%cst : f32) outs(%init : tensor<?x?xf32>) -> tensor<?x?xf32>
  %1 = linalg.matmul {lowering_config = #iree_codegen.lowering_config<tile_sizes = [[10, 20, 30]]>}
      ins(%arg0, %arg1 : tensor<?x?xf32>, tensor<?x?xf32>)
      outs(%0 : tensor<?x?xf32>) -> tensor<?x?xf32>
  %2 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1)-> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]}
    ins(%1, %arg2 : tensor<?x?xf32>, tensor<?xf32>)
    outs(%init : tensor<?x?xf32>) {
      ^bb0(%arg3: f32, %arg4: f32, %arg5: f32):
        %3 = arith.addf %arg3, %arg4 : f32
        linalg.yield %3 : f32
    } -> tensor<?x?xf32>
  return %2 : tensor<?x?xf32>
}
//      CHECK: func.func @matmul_bias_add(
//      CHECK:   scf.for
// CHECK-SAME:   {
//      CHECK:     scf.for
// CHECK-SAME:     {
//      CHECK:       linalg.fill
//      CHECK:       linalg.matmul
//      CHECK:       linalg.generic
//      CHECK:     }
//      CHECK:   }

// -----

func.func @all_zeros(%arg0 : tensor<?x?xf32>, %arg1 : tensor<?x?xf32>, %arg2 : tensor<?xf32>) -> tensor<?x?xf32> {
  %cst = arith.constant 0.0 : f32
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %d0 = tensor.dim %arg0, %c0 : tensor<?x?xf32>
  %d1 = tensor.dim %arg1, %c1 : tensor<?x?xf32>
  %init = tensor.empty(%d0, %d1) : tensor<?x?xf32>
  %0 = linalg.fill ins(%cst : f32) outs(%init : tensor<?x?xf32>) -> tensor<?x?xf32>
  %1 = linalg.matmul {lowering_config = #iree_codegen.lowering_config<tile_sizes = [[0, 0, 0]]>}
      ins(%arg0, %arg1 : tensor<?x?xf32>, tensor<?x?xf32>)
      outs(%0 : tensor<?x?xf32>) -> tensor<?x?xf32>
  %2 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1)-> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]}
    ins(%1, %arg2 : tensor<?x?xf32>, tensor<?xf32>)
    outs(%init : tensor<?x?xf32>) {
      ^bb0(%arg3: f32, %arg4: f32, %arg5: f32):
        %3 = arith.addf %arg3, %arg4 : f32
        linalg.yield %3 : f32
    } -> tensor<?x?xf32>
  return %2 : tensor<?x?xf32>
}
//      CHECK: func.func @all_zeros(
//  CHECK-NOT:   scf.for
//      CHECK:   linalg.fill
//      CHECK:   linalg.matmul
//      CHECK:   linalg.generic

// -----

func.func @multi_config(%arg0 : tensor<?x?xf32>, %arg1 : tensor<?x?xf32>, %arg2 : tensor<?xf32>) -> tensor<?x?xf32> {
  %cst = arith.constant 0.0 : f32
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %d0 = tensor.dim %arg0, %c0 : tensor<?x?xf32>
  %d1 = tensor.dim %arg1, %c1 : tensor<?x?xf32>
  %init = tensor.empty(%d0, %d1) : tensor<?x?xf32>
  %0 = linalg.fill ins(%cst : f32) outs(%init : tensor<?x?xf32>) -> tensor<?x?xf32>
  %1 = linalg.matmul {lowering_config = #iree_codegen.lowering_config<tile_sizes = [[0, 0]]>}
      ins(%arg0, %arg1 : tensor<?x?xf32>, tensor<?x?xf32>)
      outs(%0 : tensor<?x?xf32>) -> tensor<?x?xf32>
  %2 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1)-> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]}
    ins(%1, %arg2 : tensor<?x?xf32>, tensor<?xf32>)
    outs(%init : tensor<?x?xf32>) attrs = {lowering_config = #iree_codegen.lowering_config<tile_sizes = [[10, 20]]>} {
      ^bb0(%arg3: f32, %arg4: f32, %arg5: f32):
        %3 = arith.addf %arg3, %arg4 : f32
        linalg.yield %3 : f32
    } -> tensor<?x?xf32>
  return %2 : tensor<?x?xf32>
}
// Both linalg.matmul op and linalg.generic op have lowering_config. Test that
// the lowering_config of linalg.generic op is picked in the pass. In this case,
// scf.for ops are created. If the lowering_config of linalg.matmul op is
// picked, there are no scf.for ops. Because the tiling sizes are zeros.
//      CHECK: func.func @multi_config(
//      CHECK:   scf.for
//      CHECK:     scf.for
//  CHECK-NOT:       scf.for
//      CHECK:       linalg.fill
//      CHECK:       linalg.matmul
//      CHECK:       linalg.generic

// -----

func.func @shared_out_operand() {
  %cst = arith.constant 0.000000e+00 : f32
  %cst_0 = arith.constant 6.000000e+00 : f32
  %c600576 = arith.constant 600576 : index
  %c0 = arith.constant 0 : index
  %0 = hal.interface.constant.load[0] : i32
  %1 = hal.interface.constant.load[1] : i32
  %2 = arith.index_castui %0 {stream.alignment = 1024 : index, stream.values = [205824 : index, 795648 : index, 1385472 : index, 1975296 : index, 2565120 : index, 3154944 : index, 3744768 : index]} : i32 to index
  %3 = arith.index_castui %1 {stream.alignment = 1024 : index, stream.values = [0 : index, 3072 : index, 6144 : index, 9216 : index, 12288 : index, 15360 : index, 18432 : index]} : i32 to index
  %4 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<391x384xf32>>
  %5 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%2) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<384x384xf32>>
  %6 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%3) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<384xf32>>
  %7 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) alignment(64) offset(%c600576) : !flow.dispatch.tensor<writeonly:tensor<391x384xf32>>
  %8 = flow.dispatch.tensor.load %4, offsets = [0, 0], sizes = [391, 384], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<391x384xf32>> -> tensor<391x384xf32>
  %9 = flow.dispatch.tensor.load %5, offsets = [0, 0], sizes = [384, 384], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<384x384xf32>> -> tensor<384x384xf32>
  %10 = flow.dispatch.tensor.load %6, offsets = [0], sizes = [384], strides = [1] : !flow.dispatch.tensor<readonly:tensor<384xf32>> -> tensor<384xf32>
  %11 = flow.dispatch.tensor.load %7, offsets = [0, 0], sizes = [391, 384], strides = [1, 1] : !flow.dispatch.tensor<writeonly:tensor<391x384xf32>> -> tensor<391x384xf32>
  %12 = linalg.fill ins(%cst : f32) outs(%11 : tensor<391x384xf32>) -> tensor<391x384xf32>
  %13 = linalg.matmul {lowering_config = #iree_codegen.lowering_config<tile_sizes = [[8, 128, 0]]>}
    ins(%8, %9 : tensor<391x384xf32>, tensor<384x384xf32>)
    outs(%12 : tensor<391x384xf32>) -> tensor<391x384xf32>
  %14 = linalg.generic {
      indexing_maps = [affine_map<(d0, d1) -> (d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
      iterator_types = ["parallel", "parallel"]
    }
    ins(%10, %13 : tensor<384xf32>, tensor<391x384xf32>)
    outs(%11 : tensor<391x384xf32>) {
  ^bb0(%in: f32, %in_1: f32, %out: f32):
    %15 = arith.addf %in, %in_1 : f32
    %16 = arith.minimumf %15, %cst_0 : f32
    %17 = arith.maximumf %16, %cst : f32
    linalg.yield %17 : f32
  } -> tensor<391x384xf32>
  flow.dispatch.tensor.store %14, %7, offsets = [0, 0], sizes = [391, 384], strides = [1, 1] : tensor<391x384xf32> -> !flow.dispatch.tensor<writeonly:tensor<391x384xf32>>
  return
}
//      CHECK: func.func @shared_out_operand(
//  CHECK-DAG:   %[[CST0:.+]] = arith.constant 0.000000e+00 : f32
//  CHECK-DAG:   %[[DST_BINDING:.+]] = hal.interface.binding.subspan {{.+}} : !flow.dispatch.tensor<writeonly:tensor<391x384xf32>>
//      CHECK:   %[[DST:.+]] = flow.dispatch.tensor.load %[[DST_BINDING]]
//      CHECK:   scf.for
// CHECK-SAME:       iter_args(%[[ITER0:.+]] = %[[DST]])
//      CHECK:     scf.for
// CHECK-SAME:       iter_args(%[[ITER1:.+]] = %[[ITER0]])
//      CHECK:       %[[OUT_SLICE:.+]] = tensor.extract_slice %[[ITER1]]
//      CHECK:       %{{.+}} = linalg.fill ins(%[[CST0]] : f32) outs(%[[OUT_SLICE]]
//      CHECK:       %{{.+}} = linalg.matmul
//      CHECK:       %[[OUT_SLICE2:.+]] = tensor.extract_slice %[[ITER1]]
//      CHECK:       %{{.+}} = linalg.generic
// CHECK-SAME:         outs(%[[OUT_SLICE2]]

// -----

// This test is to check it doesnt crash. See #15126
func.func @softmax() {
  %c2 = arith.constant 2 : index
  %c5 = arith.constant 5 : index
  %cst = arith.constant 0xFF800000 : f32
  %c10 = arith.constant 10 : index
  %c1 = arith.constant 1 : index
  %cst_0 = arith.constant 0.000000e+00 : f32
  %cst_1 = arith.constant -1.000000e+30 : f32
  %c512 = arith.constant 512 : index
  %c0 = arith.constant 0 : index
  %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c512) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<1x10xf32>>
  %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<writeonly:tensor<1x10xf32>>
  %2 = flow.dispatch.tensor.load %1, offsets = [0, 0], sizes = [1, 10], strides = [1, 1] : !flow.dispatch.tensor<writeonly:tensor<1x10xf32>> -> tensor<1x10xf32>
  %3 = flow.dispatch.tensor.load %0, offsets = [0, 0], sizes = [1, 10], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<1x10xf32>> -> tensor<1x10xf32>
  %4 = tensor.empty() : tensor<1xf32>
  %5 = linalg.fill {lowering_config = #iree_codegen.lowering_config<tile_sizes = [[0], [0], [0], [0]]>} ins(%cst_1 : f32) outs(%4 : tensor<1xf32>) -> tensor<1xf32>
  %expanded = tensor.expand_shape %3 [[0], [1, 2]] : tensor<1x10xf32> into tensor<1x5x2xf32>
  %6 = tensor.empty() : tensor<1x2xf32>
  %7 = linalg.fill ins(%cst : f32) outs(%6 : tensor<1x2xf32>) -> tensor<1x2xf32>
  %8 = scf.for %arg0 = %c0 to %c5 step %c1 iter_args(%arg1 = %7) -> (tensor<1x2xf32>) {
    %extracted_slice = tensor.extract_slice %expanded[0, %arg0, 0] [1, 1, 2] [1, 1, 1] : tensor<1x5x2xf32> to tensor<1x1x2xf32>
    %13 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d2)>], iterator_types = ["parallel", "reduction", "parallel"]} ins(%extracted_slice : tensor<1x1x2xf32>) outs(%arg1 : tensor<1x2xf32>) {
    ^bb0(%in: f32, %out: f32):
      %14 = arith.maximumf %in, %out : f32
      linalg.yield %14 : f32
    } -> tensor<1x2xf32>
    scf.yield %13 : tensor<1x2xf32>
  }
  %9 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>], iterator_types = ["parallel", "reduction"]} ins(%8 : tensor<1x2xf32>) outs(%5 : tensor<1xf32>) {
  ^bb0(%in: f32, %out: f32):
    %13 = arith.maximumf %in, %out : f32
    linalg.yield %13 : f32
  } -> tensor<1xf32>
  %10 = linalg.fill {lowering_config = #iree_codegen.lowering_config<tile_sizes = [[0], [0], [0], [0]]>} ins(%cst_0 : f32) outs(%4 : tensor<1xf32>) -> tensor<1xf32>
  %11 = scf.for %arg0 = %c0 to %c10 step %c2 iter_args(%arg1 = %10) -> (tensor<1xf32>) {
    %extracted_slice = tensor.extract_slice %3[0, %arg0] [1, 2] [1, 1] : tensor<1x10xf32> to tensor<1x2xf32>
    %13 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>, affine_map<(d0, d1) -> (d0)>], iterator_types = ["parallel", "reduction"]} ins(%extracted_slice, %9 : tensor<1x2xf32>, tensor<1xf32>) outs(%arg1 : tensor<1xf32>) attrs =  {lowering_config = #iree_codegen.lowering_config<tile_sizes = [[0, 0], [0, 0], [0, 2], [0, 0]]>} {
    ^bb0(%in: f32, %in_2: f32, %out: f32):
      %14 = arith.subf %in, %in_2 : f32
      %15 = math.exp %14 : f32
      %16 = arith.addf %15, %out : f32
      linalg.yield %16 : f32
    } -> tensor<1xf32>
    scf.yield %13 : tensor<1xf32>
  }
  %12 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>, affine_map<(d0, d1) -> (d0)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%3, %9, %11 : tensor<1x10xf32>, tensor<1xf32>, tensor<1xf32>) outs(%2 : tensor<1x10xf32>) attrs =  {lowering_config = #iree_codegen.lowering_config<tile_sizes = [[0, 0], [0, 0], [0, 0], [2, 2]]>} {
  ^bb0(%in: f32, %in_2: f32, %in_3: f32, %out: f32):
    %13 = arith.subf %in, %in_2 : f32
    %14 = math.exp %13 : f32
    %15 = arith.divf %14, %in_3 : f32
    linalg.yield %15 : f32
  } -> tensor<1x10xf32>
  flow.dispatch.tensor.store %12, %1, offsets = [0, 0], sizes = [1, 10], strides = [1, 1] : tensor<1x10xf32> -> !flow.dispatch.tensor<writeonly:tensor<1x10xf32>>
  return
}
// CHECK-LABEL: func @softmax()

// -----

func.func @scalable_matmul(%A: tensor<?x?xf32>, %B: tensor<?x?xf32>, %C: tensor<?x?xf32>) -> tensor<?x?xf32>{
  // Matrix multiplication (ijk) with scalable tiling in the j-th dimension.
  %1 = linalg.matmul {lowering_config = #iree_codegen.lowering_config<tile_sizes = [[1, [32], 1]]>} ins(%A, %B: tensor<?x?xf32>, tensor<?x?xf32>)
            outs(%C: tensor<?x?xf32>) -> tensor<?x?xf32>
  return %1 : tensor<?x?xf32>
}
// CHECK-LABEL: func.func @scalable_matmul(
//   CHECK-DAG: %[[C1:.*]] = arith.constant 1 : index
//   CHECK-DAG: %[[C32:.*]] = arith.constant 32 : index
//       CHECK: %[[VSCALE:.*]] = vector.vscale
//  CHECK-NEXT: %[[SCALABLE_TILE_SIZE:.*]] = arith.muli %[[VSCALE]], %[[C32]] : index
//       CHECK: scf.for
//  CHECK-SAME:     step %[[C1]]
//       CHECK:   scf.for
//  CHECK-SAME:       step %[[SCALABLE_TILE_SIZE]]
//       CHECK:     scf.for
//  CHECK-SAME:         step %[[C1]]

// -----

#map = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
func.func @ukernel_generic(%arg0: tensor<1x192x1x16xf32>, %arg1: tensor<1x768x1x1xf32>, %arg2: tensor<192x768x16x1xf32>, %arg3: tensor<1x192x1x16xf32>) -> tensor<1x192x1x16xf32> {
  %c1 = arith.constant 1 : index
  %c192 = arith.constant 192 : index
  %c768 = arith.constant 768 : index
  %c1_i32 = arith.constant 1 : i32
  %c16_i32 = arith.constant 16 : i32
  %c1025_i32 = arith.constant 1025 : i32
  %0 = tensor.empty() : tensor<1x192x1x16xf32>
  %1 = iree_codegen.ukernel.generic "iree_uk_mmt4d" ins(%arg1, %arg2 : tensor<1x768x1x1xf32>, tensor<192x768x16x1xf32>) outs(%0 : tensor<1x192x1x16xf32>) (%c1, %c192, %c768, %c1_i32, %c16_i32, %c1_i32, %c1025_i32 : index, index, index, i32, i32, i32, i32) fn_def_attrs {hal.import.bitcode = true, hal.import.cconv = 1 : i32, hal.import.fields = ["processor_data"]} strided_outer_dims(1) -> tensor<1x192x1x16xf32>
  %2 = linalg.generic { indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"] }
    ins(%1, %arg3 : tensor<1x192x1x16xf32>, tensor<1x192x1x16xf32>)
    outs(%arg0 : tensor<1x192x1x16xf32>)
    attrs = {lowering_config = #iree_codegen.lowering_config<tile_sizes = [[0, 20, 0]]>} {
  ^bb0(%in: f32, %in_0: f32, %out: f32):
    %3 = arith.addf %in, %in_0 : f32
    linalg.yield %3 : f32
  } -> tensor<1x192x1x16xf32>
  return %2 : tensor<1x192x1x16xf32>
}
// CHECK-LABEL: func.func @ukernel_generic
// CHECK-SAME:    %[[ARG0:[a-zA-Z0-9]+]]
// CHECK-SAME:    %[[ARG1:[a-zA-Z0-9]+]]
// CHECK-SAME:    %[[ARG2:[a-zA-Z0-9]+]]
// CHECK-SAME:    %[[ARG3:[a-zA-Z0-9]+]]
// CHECK:         %[[UK:.+]] = iree_codegen.ukernel.generic "iree_uk_mmt4d"
// CHECK:         scf.for {{.+}} iter_args(%[[ITER:.+]] = %[[ARG0]])
// CHECK:           %[[UK_SLICE:.+]] = tensor.extract_slice %[[UK]]
// CHECK:           %[[ARG3_SLICE:.+]] = tensor.extract_slice %[[ARG3]]
// CHECK:           %[[ITER_SLICE:.+]] = tensor.extract_slice %[[ITER]]
// CHECK:           linalg.generic
// CHECK-SAME:        ins(%[[UK_SLICE]], %[[ARG3_SLICE]]
// CHECK-SAME:        outs(%[[ITER_SLICE]]

// RUN: iree-opt --pass-pipeline="builtin.module(func.func(iree-codegen-gpu-tile-reduction),canonicalize,cse)" --split-input-file %s | FileCheck %s

func.func @warp_reduction_dispatch() {
  %cst = arith.constant 1.000000e+00 : f32
  %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) : !flow.dispatch.tensor<readonly:tensor<512x10240xf32>>
  %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) : !flow.dispatch.tensor<writeonly:tensor<512xf32>>
  %workgroup_id_x = hal.interface.workgroup.id[0] : index
  %2 = flow.dispatch.tensor.load %1, offsets = [%workgroup_id_x], sizes = [1], strides = [1] : !flow.dispatch.tensor<writeonly:tensor<512xf32>> -> tensor<1xf32>
  %3 = flow.dispatch.tensor.load %0, offsets = [%workgroup_id_x, 0], sizes = [1, 10240], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<512x10240xf32>> -> tensor<1x10240xf32>
  %4 = linalg.fill {lowering_config = #iree_codegen.lowering_config<tile_sizes = [[1], [0, 2048]]>} ins(%cst : f32) outs(%2 : tensor<1xf32>) -> tensor<1xf32>
  %5 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>],
    iterator_types = ["parallel", "reduction"]}
    ins(%3 : tensor<1x10240xf32>) outs(%4 : tensor<1xf32>)
    attrs =  {lowering_config = #iree_codegen.lowering_config<tile_sizes = [[1], [0, 2048]]>} {
  ^bb0(%in: f32, %out: f32):
    %6 = arith.addf %in, %out : f32
    linalg.yield %6 : f32
  } -> tensor<1xf32>
  flow.dispatch.tensor.store %5, %1, offsets = [%workgroup_id_x], sizes = [1], strides = [1] : tensor<1xf32> -> !flow.dispatch.tensor<writeonly:tensor<512xf32>>
  return
}

//   CHECK-DAG: #[[$MAP0:.+]] = affine_map<(d0, d1) -> (d0, d1)>
//   CHECK-DAG: #[[$MAP1:.+]] = affine_map<(d0, d1) -> (d0)>
// CHECK-LABEL: warp_reduction_dispatch()
//   CHECK-DAG:   %[[C0:.+]] = arith.constant 0 : index
//   CHECK-DAG:   %[[C2048:.+]] = arith.constant 2048 : index
//   CHECK-DAG:   %[[C10240:.+]] = arith.constant 10240 : index
//   CHECK-DAG:   %[[IDEN:.+]] = arith.constant 0.000000e+00 : f32
//       CHECK:   %[[F0:.+]] = linalg.fill
//       CHECK:   %[[F1:.+]] = linalg.fill ins(%[[IDEN]] : f32) outs(%{{.*}} : tensor<1x2048xf32>) -> tensor<1x2048xf32>
//       CHECK:   %[[A1:.*]] = scf.for %[[IV:.+]] = %[[C0]] to %[[C10240]] step %[[C2048]] iter_args(%[[A0:.+]] = %[[F1]]) -> (tensor<1x2048xf32>) {
//       CHECK:     %[[S:.+]] = tensor.extract_slice %{{.*}}[0, %[[IV]]] [1, 2048] [1, 1] : tensor<1x10240xf32> to tensor<1x2048xf32>
//       CHECK:     %[[A2:.+]] = linalg.generic {indexing_maps = [#[[$MAP0]], #[[$MAP0]]], iterator_types = ["parallel", "parallel"]} ins(%[[S]] : tensor<1x2048xf32>) outs(%[[A0]] : tensor<1x2048xf32>) {
//       CHECK:       arith.addf {{.*}} : f32
//       CHECK:     } -> tensor<1x2048xf32>
//       CHECK:     scf.yield %[[A2]] : tensor<1x2048xf32>
//       CHECK:   }
//       CHECK:   %[[A3:.+]] = linalg.generic {indexing_maps = [#[[$MAP0]], #[[$MAP1]]], iterator_types = ["parallel", "reduction"]} ins(%[[A1]] : tensor<1x2048xf32>) outs(%[[F0]] : tensor<1xf32>) {
//       CHECK:     arith.addf %in, %out : f32
//       CHECK:   } -> tensor<1xf32>
//       CHECK:   flow.dispatch.tensor.store %[[A3]]

// -----

func.func @warp_reduction_batch_matmul() {
  %cst = arith.constant 1.000000e+00 : f32
  %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) : !flow.dispatch.tensor<readonly:tensor<11x512x512xf32>>
  %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) : !flow.dispatch.tensor<readonly:tensor<11x512x512xf32>>
  %2 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) : !flow.dispatch.tensor<writeonly:tensor<11x512x512xf32>>
  %workgroup_id_x = hal.interface.workgroup.id[0] : index
  %workgroup_id_y = hal.interface.workgroup.id[1] : index
  %workgroup_id_z = hal.interface.workgroup.id[2] : index
  %3 = flow.dispatch.tensor.load %0, offsets = [%workgroup_id_z, %workgroup_id_y, 0], sizes = [1, 1, 512], strides = [1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<11x512x512xf32>> -> tensor<1x1x512xf32>
  %4 = flow.dispatch.tensor.load %1, offsets = [%workgroup_id_z, 0, %workgroup_id_x], sizes = [1, 512, 1], strides = [1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<11x512x512xf32>> -> tensor<1x512x1xf32>
  %5 = flow.dispatch.tensor.load %2, offsets = [%workgroup_id_z, %workgroup_id_y, %workgroup_id_x], sizes = [1, 1, 1], strides = [1, 1, 1] : !flow.dispatch.tensor<writeonly:tensor<11x512x512xf32>> -> tensor<1x1x1xf32>
  %6 = linalg.fill {lowering_config = #iree_codegen.lowering_config<tile_sizes = [[1, 1, 1], [0, 0, 0, 64]]>} ins(%cst : f32) outs(%5 : tensor<1x1x1xf32>) -> tensor<1x1x1xf32>
  %7 = linalg.batch_matmul {lowering_config = #iree_codegen.lowering_config<tile_sizes = [[1, 1, 1], [0, 0, 0, 64]]>}
           ins(%3, %4: tensor<1x1x512xf32>, tensor<1x512x1xf32>) outs(%6: tensor<1x1x1xf32>) -> tensor<1x1x1xf32>
  flow.dispatch.tensor.store %7, %2, offsets = [%workgroup_id_z, %workgroup_id_y, %workgroup_id_x], sizes = [1, 1, 1], strides = [1, 1, 1] : tensor<1x1x1xf32> -> !flow.dispatch.tensor<writeonly:tensor<11x512x512xf32>>
  return
}

// CHECK-LABEL: warp_reduction_batch_matmul()
//   CHECK-DAG:   %[[C512:.+]] = arith.constant 512 : index
//   CHECK-DAG:   %[[C64:.+]] = arith.constant 64 : index
//   CHECK-DAG:   %[[C0:.+]] = arith.constant 0 : index
//       CHECK:   linalg.fill {{.*}} -> tensor<1x1x1xf32>
//       CHECK:   linalg.fill {{.*}} -> tensor<1x1x1x64xf32>
//       CHECK:   scf.for {{.*}} = %[[C0]] to %[[C512]] step %[[C64]] {{.*}} -> (tensor<1x1x1x64xf32>)
//       CHECK:     linalg.generic
//  CHECK-SAME:         ins({{.*}} : tensor<1x1x64xf32>, tensor<1x64x1xf32>)
//  CHECK-SAME:         outs({{.*}} : tensor<1x1x1x64xf32>)
//       CHECK:       arith.mulf
//       CHECK:       arith.addf
//       CHECK:   %[[FINAL:.+]] = linalg.generic
//  CHECK-SAME:                   ins({{.*}} : tensor<1x1x1x64xf32>)
//  CHECK-SAME:                   outs({{.*}} : tensor<1x1x1xf32>)
//       CHECK:     arith.addf
//       CHECK:   flow.dispatch.tensor.store %[[FINAL]]

// -----

func.func @warp_reduction_broadcast_dispatch() {
  %cst = arith.constant 1.000000e+00 : f32
  %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) : !flow.dispatch.tensor<readonly:tensor<512x10240xf32>>
  %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) : !flow.dispatch.tensor<writeonly:tensor<512x10240xf32>>
  %workgroup_id_x = hal.interface.workgroup.id[0] : index
  %2 = flow.dispatch.tensor.load %1, offsets = [%workgroup_id_x, 0], sizes = [1, 10240], strides = [1, 1] : !flow.dispatch.tensor<writeonly:tensor<512x10240xf32>> -> tensor<1x10240xf32>
  %3 = flow.dispatch.tensor.load %0, offsets = [%workgroup_id_x, 0], sizes = [1, 10240], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<512x10240xf32>> -> tensor<1x10240xf32>
  %e = tensor.empty() : tensor<1xf32>
  %4 = linalg.fill {lowering_config = #iree_codegen.lowering_config<tile_sizes = [[1], [0, 2048]]>} ins(%cst : f32) outs(%e : tensor<1xf32>) -> tensor<1xf32>
  %5 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>],
    iterator_types = ["parallel", "reduction"]}
    ins(%3 : tensor<1x10240xf32>) outs(%4 : tensor<1xf32>)
    attrs =  {lowering_config = #iree_codegen.lowering_config<tile_sizes = [[1], [0, 2048]]>} {
  ^bb0(%in: f32, %out: f32):
    %6 = arith.addf %in, %out : f32
    linalg.yield %6 : f32
  } -> tensor<1xf32>
  %b = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0)>, affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]}
    ins(%5 : tensor<1xf32>) outs(%2 : tensor<1x10240xf32>)
    attrs = {lowering_config = #iree_codegen.lowering_config<tile_sizes = [[], [0, 2048]]>} {
  ^bb0(%in: f32, %out: f32):
    %6 = arith.addf %in, %in : f32
    linalg.yield %6 : f32
  } -> tensor<1x10240xf32>
  flow.dispatch.tensor.store %b, %1, offsets = [%workgroup_id_x, 0], sizes = [1, 10240], strides = [1, 1] : tensor<1x10240xf32> -> !flow.dispatch.tensor<writeonly:tensor<512x10240xf32>>
  return
}

//   CHECK-DAG: #[[$MAP0:.+]] = affine_map<(d0, d1) -> (d0, d1)>
//   CHECK-DAG: #[[$MAP1:.+]] = affine_map<(d0, d1) -> (d0)>
// CHECK-LABEL: warp_reduction_broadcast_dispatch()
//   CHECK-DAG:   %[[C0:.+]] = arith.constant 0 : index
//   CHECK-DAG:   %[[C2048:.+]] = arith.constant 2048 : index
//   CHECK-DAG:   %[[C10240:.+]] = arith.constant 10240 : index
//   CHECK-DAG:   %[[IDEN:.+]] = arith.constant 0.000000e+00 : f32
//       CHECK:   %[[F0:.+]] = linalg.fill
//       CHECK:   %[[F1:.+]] = linalg.fill ins(%[[IDEN]] : f32) outs(%{{.*}} : tensor<1x2048xf32>) -> tensor<1x2048xf32>
//       CHECK:   %[[A1:.*]] = scf.for %[[IV:.+]] = %[[C0]] to %[[C10240]] step %[[C2048]] iter_args(%[[A0:.+]] = %[[F1]]) -> (tensor<1x2048xf32>) {
//       CHECK:     %[[S:.+]] = tensor.extract_slice %{{.*}}[0, %[[IV]]] [1, 2048] [1, 1] : tensor<1x10240xf32> to tensor<1x2048xf32>
//       CHECK:     %[[A2:.+]] = linalg.generic {indexing_maps = [#[[$MAP0]], #[[$MAP0]]], iterator_types = ["parallel", "parallel"]} ins(%[[S]] : tensor<1x2048xf32>) outs(%[[A0]] : tensor<1x2048xf32>) {
//       CHECK:       arith.addf {{.*}} : f32
//       CHECK:     } -> tensor<1x2048xf32>
//       CHECK:     scf.yield %[[A2]] : tensor<1x2048xf32>
//       CHECK:   }
//       CHECK:   %[[A3:.+]] = linalg.generic {indexing_maps = [#[[$MAP0]], #[[$MAP1]]], iterator_types = ["parallel", "reduction"]} ins(%[[A1]] : tensor<1x2048xf32>) outs(%[[F0]] : tensor<1xf32>) {
//       CHECK:     arith.addf %in, %out : f32
//       CHECK:   } -> tensor<1xf32>
//       CHECK:   %[[A4:.+]] = scf.for %[[IV2:.+]] = %[[C0]] to %[[C10240]] step %[[C2048]] iter_args(%[[INI:.+]] = %{{.*}}) -> (tensor<1x10240xf32>) {
//       CHECK:     %[[S2:.+]] = tensor.extract_slice %[[INI]][0, %[[IV2]]] [1, 2048] [1, 1] : tensor<1x10240xf32> to tensor<1x2048xf32>
//       CHECK:     %[[A5:.+]] = linalg.generic {indexing_maps = [#[[$MAP1]], #[[$MAP0]]], iterator_types = ["parallel", "parallel"]} ins(%[[A3]] : tensor<1xf32>) outs(%[[S2]] : tensor<1x2048xf32>)
//       CHECK:       arith.addf {{.*}} : f32
//       CHECK:     } -> tensor<1x2048xf32>
//       CHECK:     %[[I:.+]] = tensor.insert_slice %[[A5]] into %[[INI]][0, %[[IV2]]] [1, 2048] [1, 1] : tensor<1x2048xf32> into tensor<1x10240xf32>
//       CHECK:     scf.yield %[[I]] : tensor<1x10240xf32>
//       CHECK:   }
//       CHECK:   flow.dispatch.tensor.store %[[A4]]

// -----

func.func @warp_reduction_multi_reduction() {
  %cst = arith.constant 0.000000e+00 : f32
  %10 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) : !flow.dispatch.tensor<readonly:tensor<4096x86x128xi4>>
  %11 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) : !flow.dispatch.tensor<readonly:tensor<4096x86xf32>>
  %12 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) : !flow.dispatch.tensor<readonly:tensor<4096x86xf32>>
  %13 = hal.interface.binding.subspan set(0) binding(3) type(storage_buffer) : !flow.dispatch.tensor<readonly:tensor<86x128xf32>>
  %14 = hal.interface.binding.subspan set(0) binding(4) type(storage_buffer) : !flow.dispatch.tensor<writeonly:tensor<4096xf32>>
  %workgroup_id_x = hal.interface.workgroup.id[0] : index
  %15 = flow.dispatch.tensor.load %14, offsets = [%workgroup_id_x], sizes = [1], strides = [1] : !flow.dispatch.tensor<writeonly:tensor<4096xf32>> -> tensor<1xf32>
  %16 = flow.dispatch.tensor.load %13, offsets = [0, 0], sizes = [86, 128], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<86x128xf32>> -> tensor<86x128xf32>
  %17 = flow.dispatch.tensor.load %10, offsets = [%workgroup_id_x, 0, 0], sizes = [1, 86, 128], strides = [1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<4096x86x128xi4>> -> tensor<1x86x128xi4>
  %18 = flow.dispatch.tensor.load %11, offsets = [%workgroup_id_x, 0], sizes = [1, 86], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<4096x86xf32>> -> tensor<1x86xf32>
  %19 = flow.dispatch.tensor.load %12, offsets = [%workgroup_id_x, 0], sizes = [1, 86], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<4096x86xf32>> -> tensor<1x86xf32>
  %20 = linalg.fill ins(%cst : f32) outs(%15 : tensor<1xf32>) -> tensor<1xf32>
  %21 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2) -> (d1, d2)>,
                     affine_map<(d0, d1, d2) -> (d0, d1, d2)>,
                     affine_map<(d0, d1, d2) -> (d0, d1)>,
                     affine_map<(d0, d1, d2) -> (d0, d1)>,
                     affine_map<(d0, d1, d2) -> (d0)>],
    iterator_types = ["parallel", "reduction", "reduction"]
  }
  ins(%16, %17, %18, %19 : tensor<86x128xf32>, tensor<1x86x128xi4>, tensor<1x86xf32>, tensor<1x86xf32>)
  outs(%20 : tensor<1xf32>) attrs =  {lowering_config = #iree_codegen.lowering_config<tile_sizes = [[1], [0, 2, 64]]>} {
  ^bb0(%in: f32, %in_0: i4, %in_1: f32, %in_2: f32, %out: f32):
    %22 = arith.extui %in_0 : i4 to i32
    %23 = arith.uitofp %22 : i32 to f32
    %24 = arith.subf %23, %in_2 : f32
    %25 = arith.mulf %24, %in_1 : f32
    %26 = arith.mulf %in, %25 : f32
    %27 = arith.addf %26, %out : f32
    linalg.yield %27 : f32
  } -> tensor<1xf32>
  flow.dispatch.tensor.store %21, %14, offsets = [%workgroup_id_x], sizes = [1], strides = [1] : tensor<1xf32> -> !flow.dispatch.tensor<writeonly:tensor<4096xf32>>
  return
}

// CHECk-LABEL: func.func @warp_reduction_multi_reduction()

//       CHECK:  %[[FILL:.+]] = linalg.fill {{.+}} -> tensor<1x2x64xf32>

//       CHECK:  %[[LN:.+]] = scf.for %arg0 = %c0 to %c86 step %c2 iter_args(%[[ARG0:.+]] = %[[FILL]]) -> (tensor<1x2x64xf32>)
//       CHECK:    scf.for %arg2 = %c0 to %c128 step %c64 iter_args(%{{.+}} = %[[ARG0]]) -> (tensor<1x2x64xf32>)
//       CHECK:      linalg.generic
//  CHECK-SAME:        iterator_types = ["parallel", "parallel", "parallel"]
//       CHECK:      scf.yield %{{.+}} : tensor<1x2x64xf32>
//       CHECK:    scf.yield %{{.+}} : tensor<1x2x64xf32>

//       CHECK:  linalg.generic
//  CHECK-SAME:    iterator_types = ["parallel", "reduction", "reduction"]
//  CHECK-SAME:    ins(%[[LN]] : tensor<1x2x64xf32>)
//  CHECK-SAME:    outs(%{{.+}} : tensor<1xf32>)


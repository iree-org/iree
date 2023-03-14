// RUN: iree-opt -iree-codegen-gpu-tile-reduction --split-input-file -canonicalize -cse %s | FileCheck %s

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

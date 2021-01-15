// RUN: iree-opt -split-input-file -iree-flow-dispatch-id-canonicalizations %s | IreeFileCheck %s

#map0 = affine_map<(d0) -> (d0 * 2)>

// CHECK-LABEL: func @single_iter_for
//   CHECK-NOT:   scf.for
//       CHECK:   flow.return
func @single_iter_for(%arg0: tensor<8x8xf32>) -> tensor<8x8xf32> {
  %c4 = constant 4 : index
  %3 = flow.dispatch.workgroups[%c4, %c4, %c4] (%arg0) : (tensor<8x8xf32>)
    -> tensor<8x8xf32> = (%arg1 : !flow.dispatch.input<8x8xf32>, %arg2 : !flow.dispatch.output<8x8xf32>) {
    %c1 = constant 1 : index
    %c2 = constant 2 : index
    %c8 = constant 8 : index
    %workgroup_id_2 = flow.dispatch.workgroup.id[2] : index
    %workgroup_count_2 = flow.dispatch.workgroup.count[2] : index
    %4 = affine.apply #map0(%workgroup_id_2)
    %5 = affine.apply #map0(%workgroup_count_2)
    scf.for %arg4 = %4 to %c8 step %5 {
      %12 = flow.dispatch.input.load %arg1, offsets = [%arg4, %arg4], sizes = [%c2, %c2], strides = [%c1, %c1] : !flow.dispatch.input<8x8xf32> -> tensor<?x?xf32>
      flow.dispatch.output.store %12, %arg2, offsets = [%arg4, %arg4], sizes = [%c2, %c2], strides = [%c1, %c1] : tensor<?x?xf32> -> !flow.dispatch.output<8x8xf32>
    }
    flow.return
  }
  return %3 : tensor<8x8xf32>
}

// -----

#map0 = affine_map<(d0) -> (2 * d0 + 3)>

// CHECK-LABEL: func @zero_or_one_iter_for
//       CHECK:   scf.for
//       CHECK:   flow.return
func @zero_or_one_iter_for(%arg0: tensor<8x8xf32>) -> tensor<8x8xf32> {
  %c4 = constant 4 : index
  %3 = flow.dispatch.workgroups[%c4, %c4, %c4] (%arg0) : (tensor<8x8xf32>)
    -> tensor<8x8xf32> = (%arg1 : !flow.dispatch.input<8x8xf32>, %arg2 : !flow.dispatch.output<8x8xf32>) {
    %c1 = constant 1 : index
    %c2 = constant 2 : index
    %c8 = constant 8 : index
    %workgroup_id_2 = flow.dispatch.workgroup.id[2] : index
    %workgroup_count_2 = flow.dispatch.workgroup.count[2] : index
    %4 = affine.apply #map0(%workgroup_id_2)
    %5 = affine.apply #map0(%workgroup_count_2)
    scf.for %arg4 = %4 to %c8 step %5 {
      %12 = flow.dispatch.input.load %arg1, offsets = [%arg4, %arg4], sizes = [%c2, %c2], strides = [%c1, %c1] : !flow.dispatch.input<8x8xf32> -> tensor<?x?xf32>
      flow.dispatch.output.store %12, %arg2, offsets = [%arg4, %arg4], sizes = [%c2, %c2], strides = [%c1, %c1] : tensor<?x?xf32> -> !flow.dispatch.output<8x8xf32>
    }
    flow.return
  }
  return %3 : tensor<8x8xf32>
}

// -----

#map0 = affine_map<(d0) -> (d0 * 2)>

// CHECK-LABEL: func @affine_min_canonicalization
//       CHECK:   %[[C2:.+]] = constant 2 : index
//   CHECK-NOT:   affine.min
//       CHECK:   flow.dispatch.input.load %{{.*}}, offsets = [%{{.*}}, %{{.*}}], sizes = [%[[C2]], %[[C2]]]
func @affine_min_canonicalization(%arg0: tensor<8x8xf32>) -> tensor<8x8xf32> {
  %c4 = constant 4 : index
  %0 = flow.dispatch.workgroups[%c4, %c4, %c4] (%arg0) : (tensor<8x8xf32>) -> tensor<8x8xf32> = (%arg1 : !flow.dispatch.input<8x8xf32>, %arg2 : !flow.dispatch.output<8x8xf32>) {
    %c1 = constant 1 : index
    %c2 = constant 2 : index
    %workgroup_id_2 = flow.dispatch.workgroup.id[2] : index
    %1 = affine.apply #map0(%workgroup_id_2)
    %2 = affine.min affine_map<(d0) -> (2, -d0 + 8)>(%1)
    %3 = flow.dispatch.input.load %arg1, offsets = [%1, %1], sizes = [%2, %2], strides = [%c1, %c1] : !flow.dispatch.input<8x8xf32> -> tensor<?x?xf32>
    flow.dispatch.output.store %3, %arg2, offsets = [%1, %1], sizes = [%2, %2], strides = [%c1, %c1] : tensor<?x?xf32> -> !flow.dispatch.output<8x8xf32>
    flow.return
  }
  return %0 : tensor<8x8xf32>
}

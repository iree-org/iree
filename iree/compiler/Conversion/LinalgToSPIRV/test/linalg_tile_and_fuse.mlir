// RUN: iree-opt -split-input-file -iree-codegen-linalg-tile-and-fuse %s | IreeFileCheck %s

module {
  // CHECK-LABEL: func @tile_only
  //  CHECK-SAME: %[[ARG0:[a-zA-Z0-9$._-]+]]: memref<4x8xi32>
  //  CHECK-SAME: %[[ARG1:[a-zA-Z0-9$._-]+]]: memref<4x8xi32>
  //  CHECK-SAME: %[[ARG2:[a-zA-Z0-9$._-]+]]: memref<4x8xi32>
  //  CHECK-SAME: local_size = dense<[32, 4, 1]>
  //       CHECK: scf.parallel
  //       CHECK:   %[[VIEW0:.+]] = subview %[[ARG0]]
  //       CHECK:   %[[VIEW1:.+]] = subview %[[ARG1]]
  //       CHECK:   %[[VIEW2:.+]] = subview %[[ARG2]]
  //       CHECK:   linalg.generic
  //  CHECK-SAME:     "workitem"
  //  CHECK-SAME:     %[[VIEW0]]
  //  CHECK-SAME:     %[[VIEW1]]
  //  CHECK-SAME:     %[[VIEW2]]
  func @tile_only(%arg0: memref<4x8xi32>, %arg1: memref<4x8xi32>,
                  %arg2: memref<4x8xi32>) {
    linalg.generic
      {args_in = 2 : i64, args_out = 1 : i64,
       indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                        affine_map<(d0, d1) -> (d0, d1)>,
                        affine_map<(d0, d1) -> (d0, d1)>],
       iterator_types = ["parallel", "parallel"]} %arg0, %arg1, %arg2 {
    ^bb0(%arg3: i32, %arg4: i32, %arg5: i32):
      %0 = addi %arg3, %arg4 : i32
      linalg.yield %0 : i32
    }: memref<4x8xi32>, memref<4x8xi32>, memref<4x8xi32>
    return
  }
}

// -----

module {
  // CHECK-LABEL: func @conv_padding
  //  CHECK-SAME: %[[ARG0:[a-zA-Z0-9$._-]+]]: memref<?x?x?x?xf32>
  //  CHECK-SAME: %[[ARG1:[a-zA-Z0-9$._-]+]]: memref<?x?x?x?xf32>
  //  CHECK-SAME: %[[ARG2:[a-zA-Z0-9$._-]+]]: memref<?x?x?x?xf32>
  //  CHECK-SAME: local_size = dense<[32, 1, 1]>
  //       CHECK: scf.parallel (%{{.+}})
  //       CHECK:   %[[VIEW1:.+]] = subview %[[ARG1]]
  //       CHECK:   %[[VIEW2:.+]] = subview %[[ARG2]]
  //       CHECK:   linalg.conv
  //  CHECK-SAME:     %[[VIEW1]]
  //  CHECK-SAME:     %[[VIEW2]]
  //  CHECK-SAME:     "workitem"
  func @conv_padding(%arg0 : memref<?x?x?x?xf32>, %arg1 : memref<?x?x?x?xf32>,
                     %arg2 : memref<?x?x?x?xf32>)
    attributes
      {iree.dispatch_fn_name = "conv_padding"} {
    linalg.conv(%arg0, %arg1, %arg2)
      {dilations = [1, 1],
       padding = dense<[[1, 1], [0, 1]]> : tensor<2x2xi64>, strides = [1, 1]} :
      memref<?x?x?x?xf32>, memref<?x?x?x?xf32>, memref<?x?x?x?xf32>
    return
  }
}

// -----

module {
  // CHECK-LABEL: func @conv_no_padding
  //  CHECK-SAME: %[[ARG0:[a-zA-Z0-9$._-]+]]: memref<?x?x?x?xf32>
  //  CHECK-SAME: %[[ARG1:[a-zA-Z0-9$._-]+]]: memref<?x?x?x?xf32>
  //  CHECK-SAME: %[[ARG2:[a-zA-Z0-9$._-]+]]: memref<?x?x?x?xf32>
  //  CHECK-SAME: local_size = dense<[32, 2, 2]>
  //       CHECK: scf.parallel (%{{.+}}, %{{.+}}, %{{.+}})
  //       CHECK:   %[[VIEW1:.+]] = subview %[[ARG1]]
  //       CHECK:   %[[VIEW2:.+]] = subview %[[ARG2]]
  //       CHECK:   linalg.conv
  //  CHECK-SAME:     %[[VIEW1]]
  //  CHECK-SAME:     %[[VIEW2]]
  //  CHECK-SAME:     "workitem"
  func @conv_no_padding(%arg0 : memref<?x?x?x?xf32>, %arg1 : memref<?x?x?x?xf32>,
                        %arg2 : memref<?x?x?x?xf32>)
    attributes
      {iree.dispatch_fn_name = "conv_no_padding"} {
    linalg.conv(%arg0, %arg1, %arg2) {dilations = [1, 1], strides = [1, 1]} :
      memref<?x?x?x?xf32>, memref<?x?x?x?xf32>, memref<?x?x?x?xf32>
    return
  }
}

// -----

#map0 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
module {
  // CHECK-LABEL: func @parallel_4D
  //       CHECK: scf.parallel (%{{.+}}, %{{.+}}, %{{.+}})
  func @parallel_4D(%arg0: memref<?x?x?x?xf32>,
                    %arg1 : memref<?x?x?x?xf32>,
                    %arg2 : memref<?x?x?x?xf32>)
  attributes {iree.dispatch_fn_name = "parallel_4D"} {
    linalg.generic
      {args_in = 2 : i64, args_out = 1 : i64,
       indexing_maps = [#map0, #map0, #map0],
       iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
      %arg0, %arg1, %arg2 {
    ^bb0(%arg3 : f32, %arg4 : f32, %arg5 : f32):
      %0 = addf %arg3, %arg4 : f32
      linalg.yield %0 : f32
    } : memref<?x?x?x?xf32>, memref<?x?x?x?xf32>, memref<?x?x?x?xf32>
    return
  }
}

// -----

func @no_tile(%arg0: memref<?x?xf32>,
              %arg1: memref<?x?xf32>,
              %ret0: memref<?x?xf32>) {
  linalg.matmul(%arg0, %arg1, %ret0) {__internal_linalg_transform__ = "no-tile"} :
    memref<?x?xf32>, memref<?x?xf32>, memref<?x?xf32>
  return
}
// CHECK-LABEL: func @no_tile
//  CHECK-SAME:   local_size = dense<[32, 4, 1]>
//   CHECK-NOT:   scf
//       CHECK:   linalg.matmul
//   CHECK-NOT:   scf
//       CHECK:   return

// -----

#map0 = affine_map<() -> ()>
#accesses = [#map0, #map0]
#trait = {
  args_in = 2 : i64,
  args_out = 1 : i64,
  indexing_maps = #accesses,
  iterator_types = []
}

func @scalar_add(%arg0 : memref<f32>, %arg1 : memref<f32>,
                 %arg2 : memref<f32>)
{
   linalg.generic #trait %arg0, %arg1, %arg2 {
   ^bb0(%arg3 : f32, %arg4 : f32, %arg5 : f32):
      %0 = addf %arg3, %arg4 : f32
      linalg.yield %0 : f32
   } : memref<f32>, memref<f32>, memref<f32>
   return
}
// CHECK-LABEL: func @scalar_add
//   CHECK-NOT:   scf.parallel
//   CHECK-NOT:   scf.for
//       CHECK:   linalg.generic
//  CHECK-SAME:     "no-tile"
//   CHECK-NOT:   scf.parallel
//   CHECK-NOT:   scf.for
//       CHECK:   return

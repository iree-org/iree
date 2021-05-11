// RUN: iree-opt -iree-codegen-fusion-of-tensor-ops -split-input-file %s | FileCheck %s

#map0 = affine_map<(d0, d1) -> (d0)>
#map1 = affine_map<(d0, d1) -> (d0, d1)>
module {
  func @fuse_load_reshape() {
    %c0 = constant 0 : index
    %0 = hal.interface.load.tensor @io::@arg0, offset = %c0 : tensor<4x25xf32>
    %1 = linalg.tensor_reshape %0 [[0, 1]] : tensor<4x25xf32> into tensor<100xf32>
    %2 = linalg.init_tensor [100, 1] : tensor<100x1xf32>
    %3 = linalg.generic {indexing_maps = [#map0, #map1],
           iterator_types = ["parallel", "parallel"]}
      ins(%1 : tensor<100xf32>) outs(%2 : tensor<100x1xf32>) {
      ^bb0(%arg0: f32, %arg1: f32):
        linalg.yield %arg0 : f32
      } -> tensor<100x1xf32>
    hal.interface.store.tensor %3, @io::@ret0, offset = %c0 : tensor<100x1xf32>
    return
  }
  hal.interface @io attributes {sym_visiblity = "private"} {
    hal.interface.binding @arg0, set=0, binding=0, type="StorageBuffer", access="Read"
    hal.interface.binding @ret0, set=0, binding=0, type="StorageBuffer", access="Write|Discard"
  }
}
// CHECK-LABEL: func @fuse_load_reshape
//       CHECK:   %[[LOAD:.+]] = hal.interface.load.tensor
//  CHECK-SAME:     tensor<100xf32>
//   CHECK-NOT:   linalg.reshape
//       CHECK:   linalg.generic
//  CHECK-SAME:     %[[LOAD]]

// -----

module {
  func @fuse_store_reshape(%arg0: tensor<100xi32>) {
    %c0 = constant 0 : index
    %0 = linalg.tensor_reshape %arg0 [[0, 1]] : tensor<100xi32> into tensor<4x25xi32>
    hal.interface.store.tensor %0, @io::@ret0, offset = %c0 : tensor<4x25xi32>
    return
  }
  hal.interface @io attributes {sym_visiblity = "private"} {
    hal.interface.binding @ret0, set=0, binding=0, type="StorageBuffer", access="Write|Discard"
  }
}

// CHECK-LABEL: func @fuse_store_reshape
//  CHECK-SAME:   %[[ARG0:[a-zA-Z0-9_]*]]: tensor<100xi32>
//       CHECK:   hal.interface.store.tensor %[[ARG0]]

// -----

#map0 = affine_map<(d0) -> (d0)>
module {
  func @example2() {
    %c0 = constant 0 : index
    %0 = hal.interface.load.tensor @io::@arg0, offset = %c0 : tensor<1x1x1x1000xf32>
    %1 = hal.interface.load.tensor @io::@arg1, offset = %c0 : tensor<1000xf32>
    %2 = linalg.init_tensor [1000] : tensor<1000xf32>
    %3 = linalg.generic {indexing_maps = [#map0, #map0], iterator_types = ["parallel"]}
      ins(%1 : tensor<1000xf32>) outs(%2 : tensor<1000xf32>) {
      ^bb0(%arg0: f32, %arg1: f32):  // no predecessors
        linalg.yield %arg0 : f32
      } -> tensor<1000xf32>
    %4 = linalg.tensor_reshape %0 [[0, 1, 2, 3]] : tensor<1x1x1x1000xf32> into tensor<1000xf32>
    %5 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel"]}
      ins(%4, %3 : tensor<1000xf32>, tensor<1000xf32>)
      outs(%2 : tensor<1000xf32>){
      ^bb0(%arg0: f32, %arg1: f32, %arg2: f32):  // no predecessors
        %7 = addf %arg0, %arg1 : f32
        linalg.yield %7 : f32
      } -> tensor<1000xf32>
    %6 = linalg.tensor_reshape %5 [[0, 1, 2, 3]] : tensor<1000xf32> into tensor<1x1x1x1000xf32>
    %7 = linalg.tensor_reshape %6 [[0, 1, 2], [3]] : tensor<1x1x1x1000xf32> into tensor<1x1000xf32>
    hal.interface.store.tensor %6, @io::@ret0, offset = %c0 : tensor<1x1x1x1000xf32>
    hal.interface.store.tensor %7, @io::@ret1, offset = %c0 : tensor<1x1000xf32>
    return
  }
  hal.interface @io attributes {sym_visibility = "private"} {
    hal.interface.binding @arg0, set=0, binding=0, type="StorageBuffer", access="Read"
    hal.interface.binding @arg1, set=0, binding=1, type="StorageBuffer", access="Read"
    hal.interface.binding @ret0, set=0, binding=2, type="StorageBuffer", access="Write|Discard"
    hal.interface.binding @ret1, set=0, binding=3, type="StorageBuffer", access="Write|Discard"
  }
}
// CHECK-LABEL: func @example2
//   CHECK-DAG:   %[[ARG0:.+]] = hal.interface.load.tensor @io::@arg0
//   CHECK-DAG:   %[[ARG1:.+]] = hal.interface.load.tensor @io::@arg1
//       CHECK:   %[[RES:.+]] = linalg.generic
//  CHECK-SAME:     %[[ARG0]], %[[ARG1]]
//   CHECK-DAG:   hal.interface.store.tensor %[[RES]]
//   CHECK-DAG:   hal.interface.store.tensor %[[RES]]

// -----

module {
  func @issue_3302() {
    %c0 = constant 0 : index
    %0 = hal.interface.load.tensor @io::@arg1, offset = %c0 : tensor<128xf32>
    %1 = linalg.init_tensor [384, 128] : tensor<384x128xf32>
    %2 = linalg.generic
      {indexing_maps = [affine_map<(d0, d1) -> (d1)>,
                        affine_map<(d0, d1) -> (d0, d1)>],
       iterator_types = ["parallel", "parallel"]}
      ins(%0 : tensor<128xf32>) outs(%1 : tensor<384x128xf32>) {
      ^bb0(%arg0: f32, %arg1: f32):  // no predecessors
        linalg.yield %arg0 : f32
      } -> tensor<384x128xf32>
    %3 = hal.interface.load.tensor @io::@arg0, offset = %c0 : tensor<384x4x32xf32>
    %4 = linalg.tensor_reshape %2 [[0], [1, 2]]
      : tensor<384x128xf32> into tensor<384x4x32xf32>
    %5 = linalg.init_tensor [4, 384, 32] : tensor<4x384x32xf32>
    %6 = linalg.generic
      {indexing_maps = [affine_map<(d0, d1, d2) -> (d1, d0, d2)>,
                        affine_map<(d0, d1, d2) -> (d1, d0, d2)>,
                        affine_map<(d0, d1, d2) -> (d0, d1, d2)>],
       iterator_types = ["parallel", "parallel", "parallel"]}
      ins(%3, %4 : tensor<384x4x32xf32>, tensor<384x4x32xf32>)
      outs(%5 : tensor<4x384x32xf32>) {
      ^bb0(%arg0: f32, %arg1: f32, %arg2: f32):  // no predecessors
        %7 = addf %arg0, %arg1 : f32
        linalg.yield %7 : f32
      } -> tensor<4x384x32xf32>
    hal.interface.store.tensor %6, @io::@ret0, offset = %c0 : tensor<4x384x32xf32>
    return
  }
  hal.interface @io attributes {sym_visibility = "private"} {
    hal.interface.binding @arg0, set=0, binding=0, type="StorageBuffer", access="Read"
    hal.interface.binding @arg1, set=0, binding=1, type="StorageBuffer", access="Read"
    hal.interface.binding @ret0, set=0, binding=2, type="StorageBuffer", access="Write|Discard"
  }
}

// CHECK-LABEL: func @issue_3302
//       CHECK:   %[[C0:.+]] = constant 0 : index
//   CHECK-DAG:   %[[ARG0:.+]] = hal.interface.load.tensor @io::@arg0, offset = %[[C0]] : tensor<384x4x32xf32>
//   CHECK-DAG:   %[[ARG1:.+]] = hal.interface.load.tensor @io::@arg1, offset = %[[C0]] : tensor<4x32xf32>
//       CHECK:   %[[T0:.+]] = linalg.generic
//  CHECK-SAME:     ins(%[[ARG0]], %[[ARG1]] : tensor<384x4x32xf32>, tensor<4x32xf32>)
//       CHECK:   hal.interface.store.tensor %[[T0]], @io::@ret0, offset = %[[C0]] : tensor<4x384x32xf32>

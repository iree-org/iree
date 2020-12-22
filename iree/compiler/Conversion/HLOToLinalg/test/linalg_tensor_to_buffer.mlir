// RUN: iree-opt -split-input-file -iree-codegen-hlo-to-linalg-on-buffers -canonicalize %s | IreeFileCheck %s

#map0 = affine_map<(d0, d1) -> (d0, d1)>

module {
  func @element_wise() {
    %c0 = constant 0 : index
    %shape = linalg.init_tensor[2, 2] : tensor<2x2xf32>
    %0 = hal.interface.load.tensor @legacy_io::@arg0, offset = %c0
      {operand_result_index = 0 : i32} : tensor<2x2xf32>
    %1 = hal.interface.load.tensor @legacy_io::@arg1, offset = %c0
      {operand_result_index = 1 : i32} : tensor<2x2xf32>
    %2 = linalg.generic {
       indexing_maps = [#map0, #map0, #map0],
       iterator_types = ["parallel", "parallel"]}
     ins(%0, %1 : tensor<2x2xf32>, tensor<2x2xf32>)
    outs(%shape : tensor<2x2xf32>) {
    ^bb0(%arg3: f32, %arg4: f32, %s: f32):       // no predecessors
      %3 = addf %arg3, %arg4 : f32
      linalg.yield %3 : f32
    } -> tensor<2x2xf32>
    hal.interface.store.tensor %2, @legacy_io::@ret0, offset = %c0
      {operand_result_index = 2 : i32} : tensor<2x2xf32>
    return
  }
  hal.interface @legacy_io attributes {sym_visibility = "private"} {
    hal.interface.binding @arg0, set=0, binding=0, type="StorageBuffer",
      access="Read"
    hal.interface.binding @arg1, set=0, binding=1, type="StorageBuffer",
      access="Read"
    hal.interface.binding @ret0, set=0, binding=2, type="StorageBuffer",
      access="Write"
  }
}
// CHECK-LABEL: func @element_wise
//   CHECK-DAG: %[[ARG2:.*]] = iree.placeholder for "interface buffer" {binding = @legacy_io::@ret0, operand_result_index = 2 : i32}
//   CHECK-DAG: %[[ARG0:.*]] = iree.placeholder for "interface buffer" {binding = @legacy_io::@arg0, operand_result_index = 0 : i32}
//   CHECK-DAG: %[[ARG1:.*]] = iree.placeholder for "interface buffer" {binding = @legacy_io::@arg1, operand_result_index = 1 : i32}
//   CHECK-NOT: hal.interface.load.tensor
//       CHECK: linalg.generic
//  CHECK-SAME:   ins(%[[ARG0]], %[[ARG1]] :
//  CHECK-SAME:   outs(%[[ARG2]] :
//       CHECK:   ^{{[a-zA-Z0-9$._-]+}}
//  CHECK-SAME:     %[[ARG3:[a-zA-Z0-9$._-]+]]: f32
//  CHECK-SAME:     %[[ARG4:[a-zA-Z0-9$._-]+]]: f32
//  CHECK-SAME:     %[[ARG5:[a-zA-Z0-9$._-]+]]: f32
//       CHECK:     addf %[[ARG3]], %[[ARG4]]
//   CHECK-NOT: hal.interface.store.tensor

// -----

#map0 = affine_map<(d0, d1) -> (d0, d1)>

module {
  func @indexed_generic() {
    %c0 = constant 0 : index
    %shape = linalg.init_tensor[2, 2] : tensor<2x2xi32>
    %0 = hal.interface.load.tensor @legacy_io::@arg0, offset = %c0
      {operand_result_index = 0 : i32} : tensor<2x2xi32>
    %1 = linalg.indexed_generic {
       indexing_maps = [#map0, #map0],
       iterator_types = ["parallel", "parallel"]}
     ins(%0 : tensor<2x2xi32>)
    outs(%shape : tensor<2x2xi32>) {
    ^bb0(%arg2: index, %arg3: index, %arg4: i32, %s: i32):       // no predecessors
      %2 = index_cast %arg2 : index to i32
      %3 = index_cast %arg3 : index to i32
      %4 = addi %arg4, %2 : i32
      %5 = addi %4, %3 : i32
      linalg.yield %5 : i32
    } -> tensor<2x2xi32>
    hal.interface.store.tensor %1, @legacy_io::@ret0, offset = %c0
      {operand_result_index = 1 : i32} : tensor<2x2xi32>
    return
  }
  hal.interface @legacy_io attributes {sym_visibility = "private"} {
    hal.interface.binding @arg0, set=0, binding=0, type="StorageBuffer",
      access="Read"
    hal.interface.binding @ret0, set=0, binding=1, type="StorageBuffer",
      access="Write"
  }
}
//      CHECK: func @indexed_generic
//  CHECK-DAG: %[[RET0:.*]] = iree.placeholder for "interface buffer" {binding = @legacy_io::@ret0, operand_result_index = 1 : i32}
//  CHECK-DAG: %[[ARG0:.*]] = iree.placeholder for "interface buffer" {binding = @legacy_io::@arg0, operand_result_index = 0 : i32}
//  CHECK-NOT: hal.interface.load.tensor
//      CHECK: linalg.indexed_generic
// CHECK-SAME:   ins(%[[ARG0]] :
// CHECK-SAME:   outs(%[[RET0]] :
//  CHECK-NOT: hal.interface.store.tensor
//      CHECK:   ^{{[a-zA-Z0-9$._-]+}}
// CHECK-SAME:       %[[ARG2:[a-zA-Z0-9$._-]+]]: index
// CHECK-SAME:       %[[ARG3:[a-zA-Z0-9$._-]+]]: index
// CHECK-SAME:       %[[ARG4:[a-zA-Z0-9$._-]+]]: i32
//      CHECK:     %[[A:.+]] = index_cast %[[ARG2]] : index to i32
//      CHECK:     %[[B:.+]] = index_cast %[[ARG3]] : index to i32
//      CHECK:     %[[C:.+]] = addi %[[ARG4]], %[[A]] : i32
//      CHECK:     %[[D:.+]] = addi %[[C]], %[[B]] : i32
//      CHECK:     linalg.yield %[[D]] : i32

// -----

#map0 = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> (d0, 0)>
#map2 = affine_map<(d0, d1) -> (0, d1)>

module {
  func @reshape_arg_result() {
    %c0 = constant 0 : index
    %shape = linalg.init_tensor[5, 5] : tensor<5x5xf32>
    %0 = hal.interface.load.tensor @legacy_io::@arg0, offset = %c0
      {operand_result_index = 0 : i32} : tensor<5xf32>
    %1 = hal.interface.load.tensor @legacy_io::@arg1, offset = %c0
      {operand_result_index = 1 : i32} : tensor<5xf32>
    %2 = linalg.tensor_reshape %0 [#map0] : tensor<5xf32> into tensor<5x1xf32>
    %3 = linalg.tensor_reshape %1 [#map0] : tensor<5xf32> into tensor<1x5xf32>
    %4 = linalg.generic {
       indexing_maps = [#map1, #map2, #map0],
       iterator_types = ["parallel", "parallel"]}
     ins(%2, %3 : tensor<5x1xf32>, tensor<1x5xf32>)
    outs(%shape : tensor<5x5xf32>) {
         ^bb0(%arg3: f32, %arg4: f32, %s: f32):       // no predecessors
           %5 = addf %arg3, %arg4 : f32
           linalg.yield %5 : f32
         } -> tensor<5x5xf32>
    %6 = linalg.tensor_reshape %4 [#map0] : tensor<5x5xf32> into tensor<25xf32>
    hal.interface.store.tensor %6, @legacy_io::@ret0, offset = %c0
      {operand_result_index = 2 : i32} : tensor<25xf32>
    return
  }
  hal.interface @legacy_io attributes {sym_visibility = "private"} {
    hal.interface.binding @arg0, set=0, binding=0,
                                 type="StorageBuffer", access="Read"
    hal.interface.binding @arg1, set=0, binding=1,
                                 type="StorageBuffer", access="Read"
    hal.interface.binding @ret0, set=0, binding=2,
                                 type="StorageBuffer", access="Write"
  }
}
//   CHECK-DAG: #[[MAP0:.*]] = affine_map<(d0, d1) -> (d0, d1)>
//   CHECK-DAG: #[[MAP1:.*]] = affine_map<(d0, d1) -> (d0, 0)>
//   CHECK-DAG: #[[MAP2:.*]] = affine_map<(d0, d1) -> (0, d1)>
//       CHECK: func @reshape_arg_result
//   CHECK-DAG:   %[[RET0:.*]] = iree.placeholder for "interface buffer" {binding = @legacy_io::@ret0, operand_result_index = 2 : i32}
//   CHECK-DAG:   %[[RESULT:.*]] = linalg.reshape %[[RET0]] [#[[MAP0]]]
//   CHECK-DAG:   %[[ARG0:.*]] = iree.placeholder for "interface buffer" {binding = @legacy_io::@arg0, operand_result_index = 0 : i32}
//   CHECK-DAG:   %[[ARG1:.*]] = iree.placeholder for "interface buffer" {binding = @legacy_io::@arg1, operand_result_index = 1 : i32}
//   CHECK-DAG:   %[[LHS:.*]] = linalg.reshape %[[ARG0]] [#[[MAP0]]]
//   CHECK-DAG:   %[[RHS:.*]] = linalg.reshape %[[ARG1]] [#[[MAP0]]]
//       CHECK:   linalg.generic
//  CHECK-SAME:     indexing_maps = [#[[MAP1]], #[[MAP2]], #[[MAP0]]]
//  CHECK-SAME:     ins(%[[LHS]], %[[RHS]] :
//  CHECK-SAME:     outs(%[[RESULT]] :

// -----

#map0 = affine_map<(d0, d1) -> (d0, d1)>
module {
  func @reshape_only() {
    %c0 = constant 0 : index
    %0 = hal.interface.load.tensor @legacy_io::@arg0, offset = %c0
      {operand_result_index = 0 : i32} : tensor<5x5xf32>
    %1 = linalg.tensor_reshape %0 [#map0] : tensor<5x5xf32> into tensor<25xf32>
    hal.interface.store.tensor %1, @legacy_io::@ret0, offset = %c0
      {operand_result_index = 1 : i32} : tensor<25xf32>
    return
  }
  hal.interface @legacy_io attributes {sym_visibility = "private"} {
    hal.interface.binding @arg0, set=0, binding=0,
                                 type="StorageBuffer", access="Read"
    hal.interface.binding @ret0, set=0, binding=1,
                                 type="StorageBuffer", access="Write"
  }
}
//       CHECK: #[[MAP0:.*]] = affine_map<(d0, d1) -> (d0, d1)>
//       CHECK: func @reshape_only
//   CHECK-DAG:   %[[RET0:.*]] = iree.placeholder for "interface buffer" {binding = @legacy_io::@ret0, operand_result_index = 1 : i32}
//   CHECK-DAG:   %[[RESULT:.*]] = linalg.reshape %[[RET0]] [#[[MAP0]]]
//   CHECK-DAG:   %[[ARG0:.*]] = iree.placeholder for "interface buffer" {binding = @legacy_io::@arg0, operand_result_index = 0 : i32}
//       CHECK:   linalg.copy(%[[ARG0]], %[[RESULT]])

// -----

#map0 = affine_map<(d0, d1) -> (d0, d1)>

module {
  func @store_value_twice() {
    %c0 = constant 0 : index
    %shape = linalg.init_tensor[2, 4] : tensor<2x4xf32>
    %0 = hal.interface.load.tensor @legacy_io::@arg0, offset = %c0
      {operand_result_index = 0 : i32} : tensor<2x4xf32>
    %1 = linalg.generic {
       indexing_maps = [#map0, #map0],
       iterator_types = ["parallel", "parallel"]}
     ins(%0 : tensor<2x4xf32>)
    outs(%shape : tensor<2x4xf32>) {
    ^bb0(%arg0: f32, %s: f32):  // no predecessors
      %2 = tanh %arg0 : f32
      linalg.yield %2 : f32
    } -> tensor<2x4xf32>
    hal.interface.store.tensor %1, @legacy_io::@ret0, offset = %c0
      {operand_result_index = 1 : i32} : tensor<2x4xf32>
    hal.interface.store.tensor %1, @legacy_io::@ret1, offset = %c0
      {operand_result_index = 2 : i32} : tensor<2x4xf32>
    return
  }
  hal.interface @legacy_io attributes {sym_visibility = "private"} {
    hal.interface.binding @arg0, set=0, binding=0, type="StorageBuffer",
      access="Read"
    hal.interface.binding @ret0, set=0, binding=1, type="StorageBuffer",
      access="Write|Discard"
    hal.interface.binding @ret1, set=0, binding=2, type="StorageBuffer",
      access="Write|Discard"
  }
}

// CHECK-LABEL: func @store_value_twice
//   CHECK-DAG:   %[[T0:.*]] = iree.placeholder for "interface buffer" {binding = @legacy_io::@ret0, operand_result_index = 1 : i32}
//       CHECK:   %[[T1:.*]] = iree.placeholder for "interface buffer" {binding = @legacy_io::@ret1, operand_result_index = 2 : i32}
//       CHECK:   %[[T2:.*]] = iree.placeholder for "interface buffer" {binding = @legacy_io::@arg0, operand_result_index = 0 : i32}
//       CHECK:   linalg.generic
//  CHECK-SAME:     ins(%[[T2]] :
//  CHECK-SAME:     outs(%[[T0]] :
//       CHECK:   linalg.copy(%[[T0]], %[[T1]])

// -----

#map0 = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1, d2) -> (d0, d1)>
#map2 = affine_map<(d0, d1, d2) -> (d2)>

module {
  func @store_reshape_src_and_result_0() {
    %c0 = constant 0 : index
    %shape = linalg.init_tensor[2, 4] : tensor<2x4xf32>
    %0 = hal.interface.load.tensor @legacy_io::@arg0, offset = %c0
      {operand_result_index = 0 : i32} : tensor<2x4xf32>
    %1 = linalg.generic {
       indexing_maps = [#map0, #map0],
       iterator_types = ["parallel", "parallel"]}
     ins(%0 : tensor<2x4xf32>)
    outs(%shape : tensor<2x4xf32>) {
    ^bb0(%arg0: f32, %s: f32):  // no predecessors
      %2 = tanh %arg0 : f32
      linalg.yield %2 : f32
    } -> tensor<2x4xf32>
    %3 = linalg.tensor_reshape %1 [#map1, #map2]
      : tensor<2x4xf32> into tensor<1x2x4xf32>
    hal.interface.store.tensor %3, @legacy_io::@ret1, offset = %c0
      {operand_result_index = 2 : i32} : tensor<1x2x4xf32>
    hal.interface.store.tensor %1, @legacy_io::@ret0, offset = %c0
      {operand_result_index = 1 : i32} : tensor<2x4xf32>
    return
  }
  hal.interface @legacy_io attributes {sym_visibility = "private"} {
    hal.interface.binding @arg0, set=0, binding=0, type="StorageBuffer",
      access="Read"
    hal.interface.binding @ret0, set=0, binding=1, type="StorageBuffer",
      access="Write|Discard"
    hal.interface.binding @ret1, set=0, binding=2, type="StorageBuffer",
      access="Write|Discard"
  }
}

// CHECK-LABEL: func @store_reshape_src_and_result_0
//       CHECK:   %[[T0:.*]] = iree.placeholder for "interface buffer" {binding = @legacy_io::@ret1, operand_result_index = 2 : i32}
//       CHECK:   %[[T1:.*]] = linalg.reshape %[[T0]]
//       CHECK:   %[[T2:.*]] = iree.placeholder for "interface buffer" {binding = @legacy_io::@ret0, operand_result_index = 1 : i32}
//       CHECK:   %[[T3:.*]] = iree.placeholder for "interface buffer" {binding = @legacy_io::@arg0, operand_result_index = 0 : i32}
//       CHECK:   linalg.generic
//  CHECK-SAME:     ins(%[[T3]] :
//  CHECK-SAME:     outs(%[[T1]] :
//       CHECK:   linalg.copy(%[[T1]], %[[T2]])
//       CHECK:   return

// -----

#map0 = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1, d2) -> (d0, d1)>
#map2 = affine_map<(d0, d1, d2) -> (d2)>

module {
  func @store_reshape_src_and_result_1() {
    %c0 = constant 0 : index
    %shape = linalg.init_tensor[2, 4] : tensor<2x4xf32>
    %0 = hal.interface.load.tensor @legacy_io::@arg0, offset = %c0
      {operand_result_index = 0 : i32} : tensor<2x4xf32>
    %1 = linalg.generic {
       indexing_maps = [#map0, #map0],
       iterator_types = ["parallel", "parallel"]}
     ins(%0 : tensor<2x4xf32>)
    outs(%shape : tensor<2x4xf32>) {
    ^bb0(%arg0: f32, %s: f32):  // no predecessors
      %2 = tanh %arg0 : f32
      linalg.yield %2 : f32
    } -> tensor<2x4xf32>
    %3 = linalg.tensor_reshape %1 [#map1, #map2]
      : tensor<2x4xf32> into tensor<1x2x4xf32>
    hal.interface.store.tensor %1, @legacy_io::@ret0, offset = %c0
      {operand_result_index = 1 : i32} : tensor<2x4xf32>
    hal.interface.store.tensor %3, @legacy_io::@ret1, offset = %c0
      {operand_result_index = 2 : i32} : tensor<1x2x4xf32>
    return
  }
  hal.interface @legacy_io attributes {sym_visibility = "private"} {
    hal.interface.binding @arg0, set=0, binding=0, type="StorageBuffer",
      access="Read"
    hal.interface.binding @ret0, set=0, binding=1, type="StorageBuffer",
      access="Write|Discard"
    hal.interface.binding @ret1, set=0, binding=2, type="StorageBuffer",
    access="Write|Discard"
  }
}

// CHECK-LABEL: func @store_reshape_src_and_result_1
//   CHECK-DAG:   %[[T0:.*]] = iree.placeholder for "interface buffer" {binding = @legacy_io::@ret0, operand_result_index = 1 : i32}
//   CHECK-DAG:   %[[T1:.*]] = iree.placeholder for "interface buffer" {binding = @legacy_io::@ret1, operand_result_index = 2 : i32}
//   CHECK-DAG:   %[[T2:.*]] = iree.placeholder for "interface buffer" {binding = @legacy_io::@arg0, operand_result_index = 0 : i32}
//       CHECK:   linalg.generic
//  CHECK-SAME:     ins(%[[T2]] :
//  CHECK-SAME:     outs(%[[T0]] :
//       CHECK:   %[[T3:.*]] = linalg.reshape %[[T0]]
//       CHECK:   linalg.copy(%[[T3]], %[[T1]])
//       CHECK:   return

// -----

#map0 = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1, d2) -> (d0, d1)>
#map2 = affine_map<(d0, d1, d2) -> (d2)>

module {
  func @store_reshape_src_and_result_2() {
    %c0 = constant 0 : index
    %shape = linalg.init_tensor[2, 4] : tensor<2x4xf32>
    %0 = hal.interface.load.tensor @legacy_io::@arg0, offset = %c0
      {operand_result_index = 0 : i32} : tensor<2x4xf32>
    %1 = linalg.generic {
       indexing_maps = [#map0, #map0],
       iterator_types = ["parallel", "parallel"]}
     ins(%0 : tensor<2x4xf32>)
    outs(%shape : tensor<2x4xf32>) {
    ^bb0(%arg0: f32, %s: f32):  // no predecessors
      %2 = tanh %arg0 : f32
      linalg.yield %2 : f32
    } -> tensor<2x4xf32>
    %3 = linalg.tensor_reshape %1 [#map1, #map2]
      : tensor<2x4xf32> into tensor<1x2x4xf32>
    %4 = linalg.tensor_reshape %1 [#map1, #map2]
      : tensor<2x4xf32> into tensor<1x2x4xf32>
    %5 = linalg.tensor_reshape %1 [#map1, #map2]
      : tensor<2x4xf32> into tensor<1x2x4xf32>
    hal.interface.store.tensor %3, @legacy_io::@ret0, offset = %c0
      {operand_result_index = 1 : i32} : tensor<1x2x4xf32>
    hal.interface.store.tensor %4, @legacy_io::@ret1, offset = %c0
      {operand_result_index = 2 : i32} : tensor<1x2x4xf32>
    hal.interface.store.tensor %5, @legacy_io::@ret2, offset = %c0
      {operand_result_index = 3 : i32} : tensor<1x2x4xf32>
    return
  }
  hal.interface @legacy_io attributes {sym_visibility = "private"} {
    hal.interface.binding @arg0, set=0, binding=0, type="StorageBuffer",
      access="Read"
    hal.interface.binding @ret0, set=0, binding=1, type="StorageBuffer",
      access="Write|Discard"
    hal.interface.binding @ret1, set=0, binding=2, type="StorageBuffer",
      access="Write|Discard"
    hal.interface.binding @ret2, set=0, binding=3, type="StorageBuffer",
      access="Write|Discard"
  }
}

// CHECK-LABEL: func @store_reshape_src_and_result_2
//   CHECK-DAG:   %[[T0:.*]] = iree.placeholder for "interface buffer" {binding = @legacy_io::@ret0, operand_result_index = 1 : i32}
//       CHECK:   %[[T1:.*]] = linalg.reshape %[[T0]]
//   CHECK-DAG:   %[[T2:.*]] = iree.placeholder for "interface buffer" {binding = @legacy_io::@ret1, operand_result_index = 2 : i32}
//   CHECK-DAG:   %[[T3:.*]] = iree.placeholder for "interface buffer" {binding = @legacy_io::@ret2, operand_result_index = 3 : i32}
//   CHECK-DAG:   %[[T4:.*]] = iree.placeholder for "interface buffer" {binding = @legacy_io::@arg0, operand_result_index = 0 : i32}
//       CHECK:   linalg.generic
//  CHECK-SAME:   ins(%[[T4]] :
//  CHECK-SAME:   outs(%[[T1]] :
//       CHECK:   linalg.copy(%[[T0]], %[[T2]])
//       CHECK:   linalg.copy(%[[T0]], %[[T3]])
//       CHECK:   return

// -----

#map0 = affine_map<(d0, d1, d2, d3) -> (d0, d1)>
#map1 = affine_map<(d0, d1, d2, d3) -> (d2, d3)>
#map2 = affine_map<(d0, d1) -> (d0, d1)>

module {
  func @edge_detect_sobel_operator_ex_dispatch_3() {
    %c0 = constant 0 : index
    %shape = linalg.init_tensor[128, 128] : tensor<128x128xf32>
    %0 = hal.interface.load.tensor @legacy_io::@arg0, offset = %c0
      {operand_result_index = 0 : i32} : tensor<1x128x128x1xf32>
    %1 = linalg.tensor_reshape %0 [#map0, #map1]
      : tensor<1x128x128x1xf32> into tensor<128x128xf32>
    %2 = linalg.tensor_reshape %0 [#map0, #map1]
      : tensor<1x128x128x1xf32> into tensor<128x128xf32>
    %3 = linalg.generic {
       indexing_maps = [#map2, #map2, #map2],
       iterator_types = ["parallel", "parallel"]}
     ins(%1, %2 : tensor<128x128xf32>, tensor<128x128xf32>)
    outs(%shape: tensor<128x128xf32>) {
    ^bb0(%arg0: f32, %arg1: f32, %s: f32):  // no predecessors
      %5 = mulf %arg0, %arg1 : f32
      linalg.yield %5 : f32
    } -> tensor<128x128xf32>
    %4 = linalg.tensor_reshape %3 [#map0, #map1]
      : tensor<128x128xf32> into tensor<1x128x128x1xf32>
    hal.interface.store.tensor %4, @legacy_io::@ret0, offset = %c0
      {operand_result_index = 1 : i32} : tensor<1x128x128x1xf32>
    return
  }
  hal.interface @legacy_io attributes {sym_visibility = "private"} {
    hal.interface.binding @arg0, set=0, binding=0, type="StorageBuffer",
      access="Read"
    hal.interface.binding @ret0, set=0, binding=1, type="StorageBuffer",
      access="Write|Discard"
  }
}
// CHECK-LABEL: func @edge_detect_sobel_operator_ex_dispatch_3
//       CHECK:   %[[T0:.*]] = iree.placeholder for "interface buffer" {binding = @legacy_io::@ret0, operand_result_index = 1 : i32}
//       CHECK:   %[[T1:.*]] = linalg.reshape %[[T0]]
//       CHECK:   %[[T2:.*]] = iree.placeholder for "interface buffer" {binding = @legacy_io::@arg0, operand_result_index = 0 : i32}
//       CHECK:   %[[T3:.*]] = linalg.reshape %[[T2]]
//       CHECK:   %[[T4:.*]] = linalg.reshape %[[T2]]
//       CHECK:   linalg.generic
//  CHECK-SAME: ins(%[[T3]], %[[T4]] :
//  CHECK-SAME: outs(%[[T1]] :
//       CHECK:   return

// -----

#map0 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#map1 = affine_map<(d0) -> (d0)>
#map2 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>
#map3 = affine_map<(d0, d1, d2, d3) -> (d3)>
module {
  func @generic_reshape_reshape() {
    %c0 = constant 0 : index
    %cst = constant 0.000000e+00 : f32
    %shape = linalg.init_tensor[1000] : tensor<1000xf32>
    %0 = hal.interface.load.tensor @legacy_io::@arg0, offset = %c0
      {operand_result_index = 0 : i32} : tensor<1x1x1x1000xf32>
    %1 = linalg.tensor_reshape %0 [#map0]
      : tensor<1x1x1x1000xf32> into tensor<1000xf32>
    %2 = linalg.generic {
       indexing_maps = [#map1, #map1], iterator_types = ["parallel"]}
     ins(%1 : tensor<1000xf32>)
    outs(%shape : tensor<1000xf32>) {
    ^bb0(%arg0: f32, %s: f32):  // no predecessors
      %5 = addf %arg0, %cst : f32
      linalg.yield %5 : f32
    } -> tensor<1000xf32>
    %3 = linalg.tensor_reshape %2 [#map0]
      : tensor<1000xf32> into tensor<1x1x1x1000xf32>
    %4 = linalg.tensor_reshape %3 [#map2, #map3]
      : tensor<1x1x1x1000xf32> into tensor<1x1000xf32>
    hal.interface.store.tensor %3, @legacy_io::@ret0, offset = %c0
      {operand_result_index = 1 : i32} : tensor<1x1x1x1000xf32>
    hal.interface.store.tensor %4, @legacy_io::@ret1, offset = %c0
      {operand_result_index = 2 : i32} : tensor<1x1000xf32>
    return
  }
  hal.interface @legacy_io attributes {sym_visibility = "private"} {
    hal.interface.binding @arg0, set=0, binding=0, type="StorageBuffer",
      access="Read"
    hal.interface.binding @ret0, set=0, binding=1, type="StorageBuffer",
      access="Write|Discard"
    hal.interface.binding @ret1, set=0, binding=2, type="StorageBuffer",
      access="Write|Discard"
  }
}

// CHECK-LABEL: func @generic_reshape_reshape
//       CHECK:   %[[RET0:.+]] = iree.placeholder
//  CHECK-SAME:     binding = @legacy_io::@ret0, operand_result_index = 1 : i32
//       CHECK:   %[[RET0_RESHAPE:.+]] = linalg.reshape %[[RET0]]
//  CHECK-SAME:     memref<1x1x1x1000xf32> into memref<1000xf32>
//       CHECK:   %[[RET1:.+]] = iree.placeholder for "interface buffer" {binding = @legacy_io::@ret1, operand_result_index = 2 : i32}
//       CHECK:   %[[ARG0:.+]] = iree.placeholder for "interface buffer" {binding = @legacy_io::@arg0, operand_result_index = 0 : i32}
//       CHECK:   %[[ARG0_RESHAPE:.+]] = linalg.reshape %[[ARG0]]
//  CHECK-SAME:     memref<1x1x1x1000xf32> into memref<1000xf32>
//       CHECK:   linalg.generic
//  CHECK-SAME:     ins(%[[ARG0_RESHAPE]] :
//  CHECK-SAME:     outs(%[[RET0_RESHAPE]] :
//       CHECK:   %[[RET0_RESHAPE2:.+]] = linalg.reshape %[[RET0]]
//  CHECK-SAME:     memref<1x1x1x1000xf32> into memref<1x1000xf32>
//       CHECK:   linalg.copy(%[[RET0_RESHAPE2]], %[[RET1]])

// -----

#map0 = affine_map<(d0, d1) -> (d0, d1)>
module {
  func @matmul_add() {
    %c0 = constant 0 : index
    %shape = linalg.init_tensor[32, 64] : tensor<32x64xf32>
    %0 = hal.interface.load.tensor @legacy_io::@arg0, offset = %c0
      : tensor<32x48xf32>
    %1 = hal.interface.load.tensor @legacy_io::@arg1, offset = %c0
      : tensor<48x64xf32>
    %2 = hal.interface.load.tensor @legacy_io::@arg2, offset = %c0
      : tensor<32x64xf32>
    %3 = "mhlo.dot"(%0, %1)
      : (tensor<32x48xf32>, tensor<48x64xf32>) -> tensor<32x64xf32>
    %4 = linalg.generic {
      indexing_maps = [#map0, #map0, #map0],
      iterator_types = ["parallel", "parallel"]}
       ins(%2, %3 : tensor<32x64xf32>, tensor<32x64xf32>)
      outs(%shape : tensor<32x64xf32>) {
        ^bb0(%arg0: f32, %arg1: f32, %s: f32):
          %5 = addf %arg0, %arg1 : f32
          linalg.yield %5 : f32
      } -> tensor<32x64xf32>
    hal.interface.store.tensor %4, @legacy_io::@ret0, offset = %c0
      : tensor<32x64xf32>
    return
  }
  hal.interface @legacy_io attributes {sym_visiblity = "private"} {
    hal.interface.binding @arg0, set=0, binding=0, type="StorageBuffer", access="Read"
    hal.interface.binding @arg1, set=0, binding=1, type="StorageBuffer", access="Read"
    hal.interface.binding @arg2, set=0, binding=2, type="StorageBuffer", access="Read"
    hal.interface.binding @ret0, set=0, binding=3, type="StorageBuffer", access="Write|Discard"
  }
}

// CHECK-LABEL: func @matmul_add
//   CHECK-DAG:   %[[RET0:.+]] = iree.placeholder for "interface buffer" {binding = @legacy_io::@ret0}
//   CHECK-DAG:   %[[ARG0:.+]] = iree.placeholder for "interface buffer" {binding = @legacy_io::@arg0}
//   CHECK-DAG:   %[[ARG1:.+]] = iree.placeholder for "interface buffer" {binding = @legacy_io::@arg1}
//   CHECK-DAG:   %[[ARG2:.+]] = iree.placeholder for "interface buffer" {binding = @legacy_io::@arg2}
//   CHECK-DAG:   %[[TEMP:.+]] = alloc()
//       CHECK:   linalg.fill(%[[TEMP]], %{{.+}})
//       CHECK:   linalg.matmul ins(%[[ARG0]], %[[ARG1]]
//  CHECK-SAME:     ) outs(%[[TEMP]]
//  CHECK-SAME:     )
//       CHECK:   linalg.generic
//  CHECK-SAME:     ins(%[[ARG2]], %[[TEMP]]
//  CHECK-SAME:     ) outs(%[[RET0]]
//  CHECK-SAME:     )
//       CHECK:   return


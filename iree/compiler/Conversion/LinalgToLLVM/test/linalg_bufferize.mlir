// RUN: iree-opt %s --iree-codegen-linalg-bufferize-llvm -canonicalize -cse -split-input-file | IreeFileCheck %s

// CHECK: #[[$DYN_MAP:.*]] = affine_map<(d0, d1)[s0, s1] -> (d0 * s1 + s0 + d1)>

// CHECK-LABEL: tile_from_tensor_load
func @tile_from_tensor_load() {
  %c0 = constant 0 : index
  %c2 = constant 2 : index
  %c4 = constant 4 : index
  %c1 = constant 1 : index

  // CHECK-DAG: iree.placeholder for "interface buffer" {binding = @legacy_io::@ret0} : memref<?x?xf32>
  // CHECK-DAG: iree.placeholder for "interface buffer" {binding = @legacy_io::@TENSOR_INIT} : memref<?x?xf32>
  // CHECK-DAG: iree.placeholder for "interface buffer" {binding = @legacy_io::@TENSOR_RHS} : memref<?x?xf32>
  // CHECK-DAG: iree.placeholder for "interface buffer" {binding = @legacy_io::@TENSOR_LHS} : memref<?x?xf32>
  %0 = hal.interface.workgroup.id[0] : index
  %1 = hal.interface.workgroup.id[1] : index
  scf.for %arg0 = %1 to %c2 step %c2 {
    scf.for %arg1 = %0 to %c4 step %c4 {

      // CHECK-DAG: subview %{{.*}}[%{{.*}}, 0] [1, 3] [1, 1] : memref<?x?xf32> to memref<1x3xf32, #[[$DYN_MAP]]>
      // CHECK-DAG: subview %{{.*}}[0, %{{.*}}] [3, 1] [1, 1] : memref<?x?xf32> to memref<3x1xf32, #[[$DYN_MAP]]>
      // CHECK-DAG: subview %{{.*}}[%{{.*}}, %{{.*}}] [1, 1] [1, 1] : memref<?x?xf32> to memref<1x1xf32, #[[$DYN_MAP]]>
      %2 = hal.interface.load.tensor.tile @legacy_io::@TENSOR_LHS, base_offset = %c0, offsets = [%arg0, 0], sizes = [1, 3], strides = [1, 1] : tensor<1x3xf32>
      %3 = hal.interface.load.tensor.tile @legacy_io::@TENSOR_RHS, base_offset = %c0, offsets = [0, %arg1], sizes = [3, 1], strides = [1, 1] : tensor<3x1xf32>
      %4 = hal.interface.load.tensor.tile @legacy_io::@TENSOR_INIT, base_offset = %c0, offsets = [%arg0, %arg1], sizes = [1, 1], strides = [1, 1] : tensor<1x1xf32>

      // CHECK: alloc() : memref<1x1xf32>
      // CHECK: linalg.copy(%{{.*}}, %{{.*}}) : memref<1x1xf32, #[[$DYN_MAP]]>, memref<1x1xf32>
      // CHECK: linalg.matmul ins(%{{.*}}, %{{.*}} : memref<1x3xf32, #[[$DYN_MAP]]>, memref<3x1xf32, #[[$DYN_MAP]]>) outs(%{{.*}} : memref<1x1xf32>)
      %5 = linalg.matmul ins(%2, %3 : tensor<1x3xf32>, tensor<3x1xf32>) outs(%4 : tensor<1x1xf32>)  -> tensor<1x1xf32>

      // CHECK: subview %{{.*}}[%{{.*}}, %{{.*}}] [1, 1] [1, 1] : memref<?x?xf32> to memref<1x1xf32, #[[$DYN_MAP]]>
      // CHECK: linalg.copy(%{{.*}}, %{{.*}}) : memref<1x1xf32>, memref<1x1xf32, #[[$DYN_MAP]]>
      hal.interface.store.tensor.tile %5, @legacy_io::@ret0, base_offset = %c0, offsets = [%arg0, %arg1], sizes = [%c1, %c1], strides = [%c1, %c1] : tensor<1x1xf32>
    }
  }
  return
}

hal.interface @legacy_io attributes {sym_visibility = "private"} {
  hal.interface.binding @TENSOR_LHS, set=0, binding=0, type="StorageBuffer", access="Read"
  hal.interface.binding @TENSOR_RHS, set=0, binding=1, type="StorageBuffer", access="Read"
  hal.interface.binding @TENSOR_INIT, set=0, binding=2, type="StorageBuffer", access="Read"
  hal.interface.binding @ret0, set=0, binding=3, type="StorageBuffer", access="Write|Discard"
}

// -----

#map = affine_map<(d0, d1) -> (d0, d1)>

// CHECK: #[[$DYN_MAP:.*]] = affine_map<(d0, d1)[s0, s1] -> (d0 * s1 + s0 + d1)>

// CHECK-LABEL: tile_from_pointwise_lhs
func @tile_from_pointwise_lhs() {
  %c0 = constant 0 : index
  %c2 = constant 2 : index
  %c4 = constant 4 : index
  %c1 = constant 1 : index

  // CHECK-DAG: iree.placeholder for "interface buffer" {binding = @legacy_io::@ret0} : memref<?x?xf32>
  // CHECK-DAG: iree.placeholder for "interface buffer" {binding = @legacy_io::@TENSOR_INIT} : memref<?x?xf32>
  // CHECK-DAG: iree.placeholder for "interface buffer" {binding = @legacy_io::@TENSOR_RHS} : memref<?x?xf32>
  // CHECK-DAG: iree.placeholder for "interface buffer" {binding = @legacy_io::@TENSOR_LHS} : memref<?x?xf32>
  %0 = hal.interface.workgroup.id[0] : index
  %1 = hal.interface.workgroup.id[1] : index
  scf.for %arg0 = %1 to %c2 step %c2 {
    scf.for %arg1 = %0 to %c4 step %c4 {

      // CHECK-DAG: subview %{{.*}}[%{{.*}}, 0] [1, 3] [1, 1] : memref<?x?xf32> to memref<1x3xf32, #[[$DYN_MAP]]>
      // CHECK-DAG: subview %{{.*}}[0, %{{.*}}] [3, 1] [1, 1] : memref<?x?xf32> to memref<3x1xf32, #[[$DYN_MAP]]>
      %2 = hal.interface.load.tensor.tile @legacy_io::@TENSOR_LHS, base_offset = %c0, offsets = [%arg0, 0], sizes = [1, 3], strides = [1, 1] : tensor<1x3xf32>
      %3 = hal.interface.load.tensor.tile @legacy_io::@TENSOR_RHS, base_offset = %c0, offsets = [0, %arg1], sizes = [3, 1], strides = [1, 1] : tensor<3x1xf32>

      %shape = linalg.init_tensor [1, 3] : tensor<1x3xf32>

      // CHECK: alloc() : memref<1x3xf32>
      // CHECK: linalg.generic {{.*}}ins(%{{.*}} : memref<1x3xf32, #[[$DYN_MAP]]>) outs(%{{.*}} : memref<1x3xf32>)
      %4 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]}
       ins(%2 : tensor<1x3xf32>)
      outs(%shape : tensor<1x3xf32>) {
      ^bb0(%arg2: f32, %s: f32):  // no predecessors
        linalg.yield %arg2 : f32
      } -> tensor<1x3xf32>

      // CHECK-DAG: subview %{{.*}}[%{{.*}}, %{{.*}}] [1, 1] [1, 1] : memref<?x?xf32> to memref<1x1xf32, #[[$DYN_MAP]]>
      // CHECK: alloc() : memref<1x1xf32>
      // CHECK: linalg.copy(%{{.*}}, %{{.*}}) : memref<1x1xf32, #[[$DYN_MAP]]>, memref<1x1xf32>
      // CHECK: linalg.matmul ins(%{{.*}}, %{{.*}} : memref<1x3xf32>, memref<3x1xf32, #[[$DYN_MAP]]>) outs(%{{.*}} : memref<1x1xf32>)
      %5 = hal.interface.load.tensor.tile @legacy_io::@TENSOR_INIT, base_offset = %c0, offsets = [%arg0, %arg1], sizes = [1, 1], strides = [1, 1] : tensor<1x1xf32>
      %6 = linalg.matmul ins(%4, %3 : tensor<1x3xf32>, tensor<3x1xf32>) outs(%5 : tensor<1x1xf32>)  -> tensor<1x1xf32>

      // CHECK: subview %{{.*}}[%{{.*}}, %{{.*}}] [1, 1] [1, 1] : memref<?x?xf32> to memref<1x1xf32, #[[$DYN_MAP]]>
      // CHECK: linalg.copy(%{{.*}}, %{{.*}}) : memref<1x1xf32>, memref<1x1xf32, #[[$DYN_MAP]]>
      hal.interface.store.tensor.tile %6, @legacy_io::@ret0, base_offset = %c0, offsets = [%arg0, %arg1], sizes = [%c1, %c1], strides = [%c1, %c1] : tensor<1x1xf32>
    }
  }
  return
}

hal.interface @legacy_io attributes {sym_visibility = "private"} {
  hal.interface.binding @TENSOR_LHS, set=0, binding=0, type="StorageBuffer", access="Read"
  hal.interface.binding @TENSOR_RHS, set=0, binding=1, type="StorageBuffer", access="Read"
  hal.interface.binding @TENSOR_INIT, set=0, binding=2, type="StorageBuffer", access="Read"
  hal.interface.binding @ret0, set=0, binding=3, type="StorageBuffer", access="Write|Discard"
}

// -----

#map = affine_map<(d0, d1) -> (d0, d1)>

// CHECK: #[[$DYN_MAP:.*]] = affine_map<(d0, d1)[s0, s1] -> (d0 * s1 + s0 + d1)>

// CHECK-LABEL: tile_from_pointwise_outs
func @tile_from_pointwise_outs() {
  %c0 = constant 0 : index
  %c2 = constant 2 : index
  %c4 = constant 4 : index
  %c1 = constant 1 : index

  // CHECK-DAG: iree.placeholder for "interface buffer" {binding = @legacy_io::@ret0} : memref<?x?xf32>
  // CHECK-DAG: iree.placeholder for "interface buffer" {binding = @legacy_io::@TENSOR_INIT} : memref<?x?xf32>
  // CHECK-DAG: iree.placeholder for "interface buffer" {binding = @legacy_io::@TENSOR_RHS} : memref<?x?xf32>
  // CHECK-DAG: iree.placeholder for "interface buffer" {binding = @legacy_io::@TENSOR_LHS} : memref<?x?xf32>
  %0 = hal.interface.workgroup.id[0] : index
  %1 = hal.interface.workgroup.id[1] : index
  scf.for %arg0 = %1 to %c2 step %c2 {
    scf.for %arg1 = %0 to %c4 step %c4 {

      // CHECK-DAG: subview %{{.*}}[%{{.*}}, 0] [1, 3] [1, 1] : memref<?x?xf32> to memref<1x3xf32, #[[$DYN_MAP]]>
      // CHECK-DAG: subview %{{.*}}[0, %{{.*}}] [3, 1] [1, 1] : memref<?x?xf32> to memref<3x1xf32, #[[$DYN_MAP]]>
      // CHECK-DAG: subview %{{.*}}[%{{.*}}, %{{.*}}] [1, 1] [1, 1] : memref<?x?xf32> to memref<1x1xf32, #[[$DYN_MAP]]>
      %2 = hal.interface.load.tensor.tile @legacy_io::@TENSOR_LHS, base_offset = %c0, offsets = [%arg0, 0], sizes = [1, 3], strides = [1, 1] : tensor<1x3xf32>
      %3 = hal.interface.load.tensor.tile @legacy_io::@TENSOR_RHS, base_offset = %c0, offsets = [0, %arg1], sizes = [3, 1], strides = [1, 1] : tensor<3x1xf32>
      %4 = hal.interface.load.tensor.tile @legacy_io::@TENSOR_INIT, base_offset = %c0, offsets = [%arg0, %arg1], sizes = [1, 1], strides = [1, 1] : tensor<1x1xf32>

      %shape = linalg.init_tensor [1, 1] : tensor<1x1xf32>
      // CHECK: %[[ALLOC:.*]] = alloc() : memref<1x1xf32>
      // CHECK: linalg.generic {{.*}}ins(%{{.*}} : memref<1x1xf32, #[[$DYN_MAP]]>) outs(%[[ALLOC]] : memref<1x1xf32>)
      %5 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]}
        ins(%4 : tensor<1x1xf32>)
       outs(%shape : tensor<1x1xf32>) {
      ^bb0(%arg2: f32, %s: f32):  // no predecessors
        linalg.yield %arg2 : f32
      } -> tensor<1x1xf32>

      // CHECK: linalg.matmul ins(%{{.*}}, %{{.*}} : memref<1x3xf32, #[[$DYN_MAP]]>, memref<3x1xf32, #[[$DYN_MAP]]>) outs(%[[ALLOC]] : memref<1x1xf32>)
      %6 = linalg.matmul ins(%2, %3 : tensor<1x3xf32>, tensor<3x1xf32>) outs(%5 : tensor<1x1xf32>)  -> tensor<1x1xf32>

      // CHECK: subview %{{.*}}[%{{.*}}, %{{.*}}] [1, 1] [1, 1] : memref<?x?xf32> to memref<1x1xf32, #[[$DYN_MAP]]>
      // CHECK: linalg.copy(%[[ALLOC]], %{{.*}}) : memref<1x1xf32>, memref<1x1xf32, #[[$DYN_MAP]]>
      hal.interface.store.tensor.tile %6, @legacy_io::@ret0, base_offset = %c0, offsets = [%arg0, %arg1], sizes = [%c1, %c1], strides = [%c1, %c1] : tensor<1x1xf32>
    }
  }
  return
}

hal.interface @legacy_io attributes {sym_visibility = "private"} {
  hal.interface.binding @TENSOR_LHS, set=0, binding=0, type="StorageBuffer", access="Read"
  hal.interface.binding @TENSOR_RHS, set=0, binding=1, type="StorageBuffer", access="Read"
  hal.interface.binding @TENSOR_INIT, set=0, binding=2, type="StorageBuffer", access="Read"
  hal.interface.binding @ret0, set=0, binding=3, type="StorageBuffer", access="Write|Discard"
}

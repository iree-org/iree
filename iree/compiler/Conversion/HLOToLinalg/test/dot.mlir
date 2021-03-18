// RUN: iree-opt -split-input-file -iree-codegen-hlo-to-linalg-pipeline -cse %s | IreeFileCheck %s

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
//   CHECK-DAG:   %[[TEMP:.+]] = memref.alloc()
//       CHECK:   linalg.fill(%[[TEMP]], %{{.+}})
//       CHECK:   linalg.matmul ins(%[[ARG0]], %[[ARG1]]
//  CHECK-SAME:     ) outs(%[[TEMP]]
//  CHECK-SAME:     )
//       CHECK:   linalg.generic
//  CHECK-SAME:     ins(%[[ARG2]], %[[TEMP]]
//  CHECK-SAME:     ) outs(%[[RET0]]
//  CHECK-SAME:     )
//       CHECK:   return

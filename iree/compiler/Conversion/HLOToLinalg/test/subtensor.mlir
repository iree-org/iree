// RUN: iree-opt -split-input-file -iree-codegen-hlo-to-linalg-on-buffers %s | IreeFileCheck %s

module  {
  //      CHECK: #[[MAP:.+]] = affine_map<(d0, d1) -> (d0 * 4 + d1 + 4)>
  //      CHECK: @slice_whole_stride
  //  CHECK-DAG: %[[IN:.+]] = iree.placeholder for "interface buffer" {binding = @io::@arg0} : memref<3x4xi32>
  //      CHECK: memref.subview %[[IN]][1, 0] [1, 4] [1, 1]  : memref<3x4xi32> to memref<1x4xi32, #[[MAP]]>
  //      CHECK: linalg.copy
  func @slice_whole_stride() attributes {signature = (tensor<3x4xi32>) -> tensor<1x4xi32>} {
    %c0 = constant 0 : index
    %0 = hal.interface.load.tensor @io::@arg0, offset = %c0 : tensor<3x4xi32>
    %1 = subtensor %0[1, 0] [1, 4] [1, 1] : tensor<3x4xi32> to tensor<1x4xi32>
    hal.interface.store.tensor %1, @io::@ret0, offset = %c0 : tensor<1x4xi32>
    return
  }
  hal.interface @io attributes {sym_visibility = "private"} {
    hal.interface.binding @arg0, set=0, binding=0, type="StorageBuffer", access="Read"
    hal.interface.binding @ret0, set=0, binding=1, type="StorageBuffer", access="Write"
  }
}

// -----

module  {
  //      CHECK: #[[MAP:.+]] = affine_map<(d0, d1) -> (d0 * 4 + d1 + 5)>
  //      CHECK: @slice_stride_part
  //  CHECK-DAG: %[[IN:.+]] = iree.placeholder for "interface buffer" {binding = @io::@arg0} : memref<3x4xi32>
  //       CHECK: memref.subview %[[IN]][1, 1] [1, 2] [1, 1]  : memref<3x4xi32> to memref<1x2xi32, #[[MAP]]>
  //       CHECK: linalg.copy
  func @slice_stride_part() attributes {signature = (tensor<3x4xi32>) -> tensor<1x2xi32>} {
    %c0 = constant 0 : index
    %0 = hal.interface.load.tensor @io::@arg0, offset = %c0 : tensor<3x4xi32>
    %1 = subtensor %0[1, 1] [1, 2] [1, 1] : tensor<3x4xi32> to tensor<1x2xi32>
    hal.interface.store.tensor %1, @io::@ret0, offset = %c0 : tensor<1x2xi32>
    return
  }
  hal.interface @io attributes {sym_visibility = "private"} {
    hal.interface.binding @arg0, set=0, binding=0, type="StorageBuffer", access="Read"
    hal.interface.binding @ret0, set=0, binding=1, type="StorageBuffer", access="Write"
  }
}

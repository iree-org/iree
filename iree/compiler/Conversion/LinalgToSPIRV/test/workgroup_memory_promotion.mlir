// RUN: iree-opt -split-input-file -pass-pipeline="hal.executable(hal.executable.target(iree-codegen-linalg-tile-and-fuse,canonicalize,cse))" -iree-spirv-use-workgroup-memory %s | IreeFileCheck %s

// TODO(GH-4901): Convert these tests back to use dynamic shapes when linalg on tensors becomes default.
hal.executable @matmul_tile attributes {sym_visibility = "private"} {
  hal.interface @legacy_io {
    hal.interface.binding @arg0, set=0, binding=0, type="StorageBuffer", access="Read"
    hal.interface.binding @arg1, set=0, binding=1, type="StorageBuffer", access="Read"
    hal.interface.binding @ret0, set=0, binding=2, type="StorageBuffer", access="Write|Discard"
  }
  hal.executable.target @vulkan, filter="dylib*" {
    hal.executable.entry_point @matmul_tile attributes {
      interface = @legacy_io, ordinal = 0 : i32,
      signature = (!flow.dispatch.input<25x50xf32>, !flow.dispatch.input<50x75xf32>,
        !flow.dispatch.output<25x75xf32>) -> ()}
    module attributes {
      spv.target_env =
        #spv.target_env<#spv.vce<v1.3, [Shader], [SPV_KHR_storage_buffer_storage_class]>,
                        {max_compute_workgroup_invocations = 128 : i32,
                         max_compute_workgroup_size = dense<[128, 128, 64]> : vector<3xi32>}>} {
      func @matmul_tile() {
        %0 = iree.placeholder for "interace buffer"
          {binding = @legacy_io::@arg0, operand_result_index = 0 : i32} : memref<25x50xf32>
        %1 = iree.placeholder for "interace buffer"
          {binding = @legacy_io::@arg1, operand_result_index = 1 : i32} : memref<50x75xf32>
        %2 = iree.placeholder for "interace buffer"
          {binding = @legacy_io::@ret0, operand_result_index = 2 : i32} : memref<25x75xf32>
        linalg.matmul ins(%0, %1 : memref<25x50xf32>, memref<50x75xf32>)
                     outs(%2 : memref<25x75xf32>)
        return
      }
      hal.interface @legacy_io attributes {sym_visibility = "private"} {
        hal.interface.binding @arg0, set=0, binding=0, type="StorageBuffer", access="Read"
        hal.interface.binding @arg1, set=0, binding=1, type="StorageBuffer", access="Read"
        hal.interface.binding @ret0, set=0, binding=2, type="StorageBuffer", access="Write"
      }
    }
  }
}
//       CHECK: func @matmul_tile()
//   CHECK-DAG:   %[[ARG0:.+]] = iree.placeholder for "interace buffer" {binding = @legacy_io::@arg0
//   CHECK-DAG:   %[[ARG1:.+]] = iree.placeholder for "interace buffer" {binding = @legacy_io::@arg1
//   CHECK-DAG:   %[[RET0:.+]] = iree.placeholder for "interace buffer" {binding = @legacy_io::@ret0
//   CHECK-DAG:   %[[ALLOC1:.+]] = alloc() : memref<8x32xf32, 3>
//   CHECK-DAG:   %[[ALLOC2:.+]] = alloc() : memref<32x16xf32, 3>
//       CHECK:   scf.for
//       CHECK:     %[[ARG0SV:.+]] = subview %[[ARG0]]
//       CHECK:     %[[ARG1SV:.+]] = subview %[[ARG1]]
//       CHECK:     %[[RET0SV:.+]] = subview %[[RET0]]
//       CHECK:     %[[SUBVIEW1:.+]] = subview %[[ALLOC1]]
//       CHECK:     %[[SUBVIEW2:.+]] = subview %[[ALLOC2]]
//       CHECK:     linalg.copy(%[[ARG0SV]], %[[SUBVIEW1]])
//  CHECK-SAME:       "copy_to_workgroup_memory"
//       CHECK:     linalg.copy(%[[ARG1SV]], %[[SUBVIEW2]])
//  CHECK-SAME:       "copy_to_workgroup_memory"
//       CHECK:     linalg.matmul
//  CHECK-SAME:       "workgroup_memory"
//  CHECK-SAME:       ins(%[[SUBVIEW1]], %[[SUBVIEW2]]
//  CHECK-SAME:      outs(%[[RET0SV]]

// -----

hal.executable @conv_no_padding_tile attributes {sym_visibility = "private"} {
  hal.interface @legacy_io {
    hal.interface.binding @arg0, set=0, binding=0, type="StorageBuffer", access="Read"
    hal.interface.binding @arg1, set=0, binding=1, type="StorageBuffer", access="Read"
    hal.interface.binding @ret0, set=0, binding=2, type="StorageBuffer", access="Write|Discard"
  }
  hal.executable.target @vulkan, filter="dylib*" {
    hal.executable.entry_point @conv_no_padding_tile attributes {
      interface = @legacy_io, ordinal = 0 : i32,
      signature = (!flow.dispatch.input<3x4x6x14xf32>, !flow.dispatch.input<2x16x16x6xf32>,
        !flow.dispatch.output<2x13x11x14xf32>) -> ()}
    module attributes {
      spv.target_env =
        #spv.target_env<#spv.vce<v1.3, [Shader], [SPV_KHR_storage_buffer_storage_class]>,
                        {max_compute_workgroup_invocations = 128 : i32,
                         max_compute_workgroup_size = dense<[128, 128, 64]> : vector<3xi32>}>} {
      func @conv_no_padding_tile() {
        %0 = iree.placeholder for "interace buffer"
          {binding = @legacy_io::@arg0, operand_result_index = 0 : i32} : memref<3x4x6x14xf32>
        %1 = iree.placeholder for "interace buffer"
          {binding = @legacy_io::@arg1, operand_result_index = 1 : i32} : memref<2x16x16x6xf32>
        %2 = iree.placeholder for "interace buffer"
          {binding = @legacy_io::@ret0, operand_result_index = 2 : i32} : memref<2x13x11x14xf32>
        linalg.conv(%0, %1, %2) {dilations = [1, 1], strides = [1, 1]}
          : memref<3x4x6x14xf32>, memref<2x16x16x6xf32>, memref<2x13x11x14xf32>
        return
      }
      hal.interface @legacy_io attributes {sym_visibility = "private"} {
        hal.interface.binding @arg0, set=0, binding=0, type="StorageBuffer", access="Read"
        hal.interface.binding @arg1, set=0, binding=1, type="StorageBuffer", access="Read"
        hal.interface.binding @ret0, set=0, binding=2, type="StorageBuffer", access="Write|Discard"
      }
    }
  }
}
//       CHECK: func @conv_no_padding_tile()
//   CHECK-DAG:   %[[ARG0:.+]] = iree.placeholder for "interace buffer" {binding = @legacy_io::@arg0
//   CHECK-DAG:   %[[ARG1:.+]] = iree.placeholder for "interace buffer" {binding = @legacy_io::@arg1
//   CHECK-DAG:   %[[RET0:.+]] = iree.placeholder for "interace buffer" {binding = @legacy_io::@ret0
//   CHECK-DAG:   %[[ALLOC1:.+]] = alloc() : memref<1x6x35x6xf32, 3>
//       CHECK:   %[[ARG1SV:.+]] = subview %[[ARG1]]
//       CHECK:   %[[RET0SV:.+]] = subview %[[RET0]]
//       CHECK:   %[[SUBVIEW1:.+]] = subview %[[ALLOC1]]
//       CHECK:   linalg.copy(%[[ARG1SV]], %[[SUBVIEW1]])
//  CHECK-SAME:      "copy_to_workgroup_memory"
//       CHECK:   linalg.conv(%[[ARG0]], %[[SUBVIEW1]], %[[RET0SV]])
//  CHECK-SAME:      "workgroup_memory"

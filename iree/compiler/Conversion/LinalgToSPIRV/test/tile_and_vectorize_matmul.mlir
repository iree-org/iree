// RUN: iree-opt -split-input-file -pass-pipeline="hal.executable(hal.executable.target(iree-spirv-concretize-tile-among-workgroups,iree-spirv-tile-and-vectorize-in-one-workgroup))" -iree-spirv-enable-vectorization -iree-codegen-spirv-experimental-linalg-on-tensors -canonicalize -cse %s | IreeFileCheck %s

hal.executable @matmul_static_shape_f16 attributes {sym_visibility = "private"} {
  hal.interface @legacy_io attributes {sym_visibility = "private"} {
    hal.interface.binding @arg0, set=0, binding=0, type="StorageBuffer", access="Read"
    hal.interface.binding @arg1, set=0, binding=1, type="StorageBuffer", access="Read"
    hal.interface.binding @ret0, set=0, binding=2, type="StorageBuffer", access="Write|Discard"
  }
  hal.executable.target @vulkan_spirv, filter="vulkan*" {
    hal.executable.entry_point @matmul_static_shape_f16 attributes {
      interface = @legacy_io, ordinal = 0 : index,
      signature = (!flow.dispatch.tensor<readonly:1x225x225x16xf32>, !flow.dispatch.tensor<readonly:3x3x16x32xf32>, !flow.dispatch.tensor<writeonly:1x112x112x32xf32>) -> ()}
    module attributes {spv.target_env = #spv.target_env<#spv.vce<v1.3, [Shader], [SPV_KHR_storage_buffer_storage_class]>, ARM:IntegratedGPU, {}>}  {
      func @matmul_static_shape_f16() {
        %cst = constant 0.000000e+00 : f16
        %c0 = constant 0 : index
        %c4096 = constant 4096 : index
        %0 = hal.interface.binding.subspan @legacy_io::@arg0[%c0] : memref<4096x4096xf16>
        %1 = hal.interface.binding.subspan @legacy_io::@arg1[%c0] : memref<4096x4096xf16>
        %2 = hal.interface.binding.subspan @legacy_io::@ret0[%c0] : memref<4096x4096xf16>
        %workgroup_size_x = hal.interface.workgroup.size[0] : index
        %workgroup_size_y = hal.interface.workgroup.size[1] : index
        %workgroup_id_x = hal.interface.workgroup.id[0] : index
        %workgroup_count_x = hal.interface.workgroup.count[0] : index
        %workgroup_id_y = hal.interface.workgroup.id[1] : index
        %workgroup_count_y = hal.interface.workgroup.count[1] : index
        %3 = affine.apply affine_map<()[s0, s1] -> (s0 * s1)>()[%workgroup_id_y, %workgroup_size_y]
        %4 = affine.apply affine_map<()[s0, s1] -> (s0 * s1)>()[%workgroup_count_y, %workgroup_size_y]
        scf.for %arg0 = %3 to %c4096 step %4 {
          %5 = affine.apply affine_map<()[s0, s1] -> (s0 * s1)>()[%workgroup_id_x, %workgroup_size_x]
          %6 = affine.apply affine_map<()[s0, s1] -> (s0 * s1)>()[%workgroup_count_x, %workgroup_size_x]
          scf.for %arg1 = %5 to %c4096 step %6 {
            %7 = affine.min affine_map<(d0)[s0] -> (s0, -d0 + 4096)>(%arg0)[%workgroup_size_y]
            %8 = memref.subview %0[%arg0, 0] [%7, 4096] [1, 1] : memref<4096x4096xf16> to memref<?x4096xf16, affine_map<(d0, d1)[s0] -> (d0 * 4096 + s0 + d1)>>
            %9 = affine.min affine_map<(d0)[s0] -> (s0, -d0 + 4096)>(%arg1)[%workgroup_size_x]
            %10 = memref.subview %2[%arg0, %arg1] [%7, %9] [1, 1] : memref<4096x4096xf16> to memref<?x?xf16, affine_map<(d0, d1)[s0] -> (d0 * 4096 + s0 + d1)>>
            %11 = memref.subview %1[0, %arg1] [4096, %9] [1, 1] : memref<4096x4096xf16> to memref<4096x?xf16, affine_map<(d0, d1)[s0] -> (d0 * 4096 + s0 + d1)>>
            linalg.fill(%10, %cst) {__internal_linalg_transform__ = "workgroup"} : memref<?x?xf16, affine_map<(d0, d1)[s0] -> (d0 * 4096 + s0 + d1)>>, f16
            linalg.matmul {__internal_linalg_transform__ = "workgroup"} ins(%8, %11 : memref<?x4096xf16, affine_map<(d0, d1)[s0] -> (d0 * 4096 + s0 + d1)>>, memref<4096x?xf16, affine_map<(d0, d1)[s0] -> (d0 * 4096 + s0 + d1)>>) outs(%10 : memref<?x?xf16, affine_map<(d0, d1)[s0] -> (d0 * 4096 + s0 + d1)>>)
          }
        }
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

//    CHECK-LABEL: func @matmul_static_shape_f16
//  CHECK-COUNT-16:   vector.transfer_write
//  CHECK-COUNT-16:   vector.transfer_read
//          CHECK:   %[[FOR_RES:.+]]:16 = scf.for
// CHECK-COUNT-16:     vector.transfer_read
// CHECK-COUNT-64:     vector.contract
//      CHECK:         scf.yield
//  CHECK-COUNT-16:    vector.transfer_write %[[FOR_RES]]
//          CHECK:    return

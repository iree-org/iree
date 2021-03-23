// RUN: iree-opt -split-input-file -pass-pipeline="hal.executable(hal.executable.target(iree-codegen-spirv-linalg-tile-and-distribute,iree-spirv-tile-and-vectorize-in-one-workgroup,canonicalize,cse))" -iree-spirv-enable-vectorization %s | IreeFileCheck %s

// CHECK-LABEL: func @elementwise_static_shape
//       CHECK:   vector.transfer_read %10[%c0], {{.*}} memref<4xf32, #map1>, vector<4xf32>
//       CHECK:   vector.transfer_read %11[%c0], {{.*}} memref<4xf32, #map1>, vector<4xf32>
//       CHECK:   addf %{{.*}}, %{{.*}} : vector<4xf32>
//       CHECK:   vector.transfer_write {{.*}} : vector<4xf32>, memref<4xf32
hal.executable @elementwise_static_shape attributes {sym_visibility = "private"} {
  hal.interface @legacy_io {
    hal.interface.binding @arg0, set=0, binding=0, type="StorageBuffer", access="Read"
    hal.interface.binding @arg1, set=0, binding=1, type="StorageBuffer", access="Read"
    hal.interface.binding @ret0, set=0, binding=2, type="StorageBuffer", access="Write|Discard"
  }
  hal.executable.target @vulkan, filter="dylib*" {
    hal.executable.entry_point @elementwise_static_shape attributes {
      interface = @legacy_io, ordinal = 0 : i32,
      signature = (!flow.dispatch.tensor<readonly:?xf32>,
        !flow.dispatch.tensor<readonly:?xf32>,
        !flow.dispatch.tensor<writeonly:?xf32>) -> ()}
    module attributes {
      spv.target_env =
        #spv.target_env<#spv.vce<v1.5,
          [Shader],
          []>, NVIDIA:DiscreteGPU,
          {subgroup_size = 32 : i32}>} {
      func @elementwise_static_shape()
        attributes {vkspv.num_workgroups_fn = @elementwise_static_shape__num_workgroups__} {
        %arg0 = iree.placeholder for "interface buffer"
          {binding = @legacy_io::@arg0, operand_result_num = 0 : i32} : memref<128xf32>
        %arg1 = iree.placeholder for "interface buffer"
          {binding = @legacy_io::@arg1, operand_result_num = 1 : i32} : memref<128xf32>
        %ret0 = iree.placeholder for "interface buffer"
          {binding = @legacy_io::@ret0, operand_result_num = 2 : i32} : memref<128xf32>
        linalg.generic {
          indexing_maps = [affine_map<(i) -> (i)>,
                           affine_map<(i) -> (i)>,
                           affine_map<(i) -> (i)>],
           iterator_types = ["parallel"]
          } ins(%arg0, %arg1 : memref<128xf32>, memref<128xf32>)
            outs(%ret0 : memref<128xf32>) {
              ^bb0(%a : f32, %b : f32, %c : f32):
              %add = addf %a, %b : f32
              linalg.yield %add : f32
        }
        return
      }
      func private @elementwise_static_shape__num_workgroups__
        (!shapex.ranked_shape<[4096, 4096]>, !shapex.ranked_shape<[4096, 4096]>,
         !shapex.ranked_shape<[4096, 4096]>) -> (index, index, index)
      hal.interface @legacy_io attributes {sym_visibility = "private"} {
        hal.interface.binding @arg0, set=0, binding=0, type="StorageBuffer", access="Read"
        hal.interface.binding @arg1, set=0, binding=1, type="StorageBuffer", access="Read"
        hal.interface.binding @ret0, set=0, binding=2, type="StorageBuffer", access="Write|Discard"
      }
    }
  }
}

// -----

// Negative test as we currently don't support vectorization when there is a
// transpose.
// CHECK-LABEL: func @elementwise_transpose
//   CHECK-NOT:   vector.transfer_read 
//       CHECK:   linalg.generic
hal.executable @elementwise_transpose attributes {sym_visibility = "private"} {
  hal.interface @legacy_io {
    hal.interface.binding @arg0, set=0, binding=0, type="StorageBuffer", access="Read"
    hal.interface.binding @arg1, set=0, binding=1, type="StorageBuffer", access="Read"
    hal.interface.binding @ret0, set=0, binding=2, type="StorageBuffer", access="Write|Discard"
  }
  hal.executable.target @vulkan, filter="dylib*" {
    hal.executable.entry_point @elementwise_transpose attributes {
      interface = @legacy_io, ordinal = 0 : i32,
      signature = (!flow.dispatch.tensor<readonly:?x?xf32>,
        !flow.dispatch.tensor<readonly:?xf32>,
        !flow.dispatch.tensor<writeonly:?x?xf32>) -> ()}
    module attributes {
      spv.target_env =
        #spv.target_env<#spv.vce<v1.5,
          [Shader],
          []>, NVIDIA:DiscreteGPU,
          {subgroup_size = 32 : i32}>} {
      func @elementwise_transpose()
        attributes {vkspv.num_workgroups_fn = @elementwise_transpose__num_workgroups__} {
        %arg0 = iree.placeholder for "interface buffer"
          {binding = @legacy_io::@arg0, operand_result_num = 0 : i32} : memref<128x8xf32>
        %arg1 = iree.placeholder for "interface buffer"
          {binding = @legacy_io::@arg1, operand_result_num = 1 : i32} : memref<128xf32>
        %ret0 = iree.placeholder for "interface buffer"
          {binding = @legacy_io::@ret0, operand_result_num = 2 : i32} : memref<128x8xf32>
        linalg.generic {
          indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                           affine_map<(d0, d1) -> (d0)>,
                           affine_map<(d0, d1) -> (d0, d1)>],
           iterator_types = ["parallel", "parallel"]
          } ins(%arg0, %arg1 : memref<128x8xf32>, memref<128xf32>)
            outs(%ret0 : memref<128x8xf32>) {
              ^bb0(%a : f32, %b : f32, %c : f32):
              %add = addf %a, %b : f32
              linalg.yield %add : f32
        }
        return
      }
      func private @elementwise_transpose__num_workgroups__
        (!shapex.ranked_shape<[4096, 4096]>, !shapex.ranked_shape<[4096, 4096]>,
         !shapex.ranked_shape<[4096, 4096]>) -> (index, index, index)
      hal.interface @legacy_io attributes {sym_visibility = "private"} {
        hal.interface.binding @arg0, set=0, binding=0, type="StorageBuffer", access="Read"
        hal.interface.binding @arg1, set=0, binding=1, type="StorageBuffer", access="Read"
        hal.interface.binding @ret0, set=0, binding=2, type="StorageBuffer", access="Write|Discard"
      }
    }
  }
}

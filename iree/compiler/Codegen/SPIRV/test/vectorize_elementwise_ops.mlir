// RUN: iree-opt -split-input-file -pass-pipeline="hal.executable(hal.executable.variant(iree-spirv-tile-and-vectorize,canonicalize,cse))" %s | IreeFileCheck %s

// CHECK-LABEL: func @elementwise_static_shape
//       CHECK:   vector.transfer_read %{{.+}}[%c0], {{.+}} memref<4xf32, #{{.+}}>, vector<4xf32>
//       CHECK:   vector.transfer_read %{{.+}}[%c0], {{.+}} memref<4xf32, #{{.+}}>, vector<4xf32>
//       CHECK:   addf %{{.*}}, %{{.*}} : vector<4xf32>
//       CHECK:   vector.transfer_write {{.*}} : vector<4xf32>, memref<4xf32
hal.executable @elementwise_static_shape attributes {sym_visibility = "private"} {
  hal.interface @io {
    hal.interface.binding @arg0, set=0, binding=0, type="StorageBuffer", access="Read"
    hal.interface.binding @arg1, set=0, binding=1, type="StorageBuffer", access="Read"
    hal.interface.binding @ret0, set=0, binding=2, type="StorageBuffer", access="Write|Discard"
  }
  hal.executable.variant @vulkan, filter="vulkan*" {
    hal.executable.entry_point @elementwise_static_shape attributes {
      interface = @io,
      ordinal = 0 : index
    }
    module attributes {
      spv.target_env =
        #spv.target_env<#spv.vce<v1.5,
          [Shader],
          []>, NVIDIA:DiscreteGPU,
          {subgroup_size = 32 : i32}>} {
      func @elementwise_static_shape() {
        %c0 = constant 0 : index
        %arg0 = hal.interface.binding.subspan @io::@arg0[%c0] : memref<128xf32>
        %arg1 = hal.interface.binding.subspan @io::@arg1[%c0] : memref<128xf32>
        %ret0 = hal.interface.binding.subspan @io::@ret0[%c0] : memref<128xf32>
        linalg.generic {
          __internal_linalg_transform__ = "workgroup",
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
      hal.interface @io attributes {sym_visibility = "private"} {
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
//       CHECK:   scf.for
//       CHECK:     scf.for
hal.executable @elementwise_transpose attributes {sym_visibility = "private"} {
  hal.interface @io {
    hal.interface.binding @arg0, set=0, binding=0, type="StorageBuffer", access="Read"
    hal.interface.binding @arg1, set=0, binding=1, type="StorageBuffer", access="Read"
    hal.interface.binding @ret0, set=0, binding=2, type="StorageBuffer", access="Write|Discard"
  }
  hal.executable.variant @vulkan, filter="dylib*" {
    hal.executable.entry_point @elementwise_transpose attributes {
      interface = @io,
      ordinal = 0 : index
    }
    module attributes {
      spv.target_env =
        #spv.target_env<#spv.vce<v1.5,
          [Shader],
          []>, NVIDIA:DiscreteGPU,
          {subgroup_size = 32 : i32}>} {
      func @elementwise_transpose() {
        %c0 = constant 0 : index
        %arg0 = hal.interface.binding.subspan @io::@arg0[%c0] : memref<128x8xf32>
        %arg1 = hal.interface.binding.subspan @io::@arg1[%c0] : memref<128xf32>
        %ret0 = hal.interface.binding.subspan @io::@ret0[%c0] : memref<128x8xf32>
        linalg.generic {
          __internal_linalg_transform__ = "workgroup",
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
      hal.interface @io attributes {sym_visibility = "private"} {
        hal.interface.binding @arg0, set=0, binding=0, type="StorageBuffer", access="Read"
        hal.interface.binding @arg1, set=0, binding=1, type="StorageBuffer", access="Read"
        hal.interface.binding @ret0, set=0, binding=2, type="StorageBuffer", access="Write|Discard"
      }
    }
  }
}

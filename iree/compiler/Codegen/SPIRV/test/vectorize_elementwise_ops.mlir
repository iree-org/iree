// RUN: iree-opt -split-input-file -pass-pipeline='hal.executable(hal.executable.variant(builtin.module(builtin.func(iree-spirv-tile-and-distribute,iree-spirv-vectorize))))' %s | IreeFileCheck %s

// CHECK-LABEL: func @elementwise_static_shape
//       CHECK:   vector.transfer_read %{{.+}}[%c0], {{.+}} memref<4xf32, #{{.+}}>, vector<4xf32>
//       CHECK:   vector.transfer_read %{{.+}}[%c0], {{.+}} memref<4xf32, #{{.+}}>, vector<4xf32>
//       CHECK:   addf %{{.*}}, %{{.*}} : vector<4xf32>
//       CHECK:   vector.transfer_write {{.*}} : vector<4xf32>, memref<4xf32
#config = #iree_codegen.lowering.config<tile_sizes = [[128], [4]], native_vector_size = []>
hal.executable private @elementwise_static_shape  {
  hal.interface @io {
    hal.interface.binding @arg0, set=0, binding=0, type="StorageBuffer"
    hal.interface.binding @arg1, set=0, binding=1, type="StorageBuffer"
    hal.interface.binding @ret0, set=0, binding=2, type="StorageBuffer"
  }
  hal.executable.variant @vulkan, target = #hal.executable.target<"vulkan-spirv", "vulkan-spirv-fb"> {
    hal.executable.entry_point @elementwise_static_shape attributes {
      interface = @io, ordinal = 0 : index,
      workgroup_size = [32: index, 1: index, 1: index]
    }
    builtin.module {
      func @elementwise_static_shape() {
        %c0 = arith.constant 0 : index
        %arg0 = hal.interface.binding.subspan @io::@arg0[%c0] : memref<128xf32>
        %arg1 = hal.interface.binding.subspan @io::@arg1[%c0] : memref<128xf32>
        %ret0 = hal.interface.binding.subspan @io::@ret0[%c0] : memref<128xf32>
        linalg.generic {
          lowering.config = #config,
          indexing_maps = [affine_map<(i) -> (i)>,
                           affine_map<(i) -> (i)>,
                           affine_map<(i) -> (i)>],
          iterator_types = ["parallel"]
        } ins(%arg0, %arg1 : memref<128xf32>, memref<128xf32>)
          outs(%ret0 : memref<128xf32>) {
              ^bb0(%a : f32, %b : f32, %c : f32):
              %add = arith.addf %a, %b : f32
              linalg.yield %add : f32
        }
        return
      }
      hal.interface private @io  {
        hal.interface.binding @arg0, set=0, binding=0, type="StorageBuffer"
        hal.interface.binding @arg1, set=0, binding=1, type="StorageBuffer"
        hal.interface.binding @ret0, set=0, binding=2, type="StorageBuffer"
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
#config = #iree_codegen.lowering.config<tile_sizes = [[1, 32], [1, 1]], native_vector_size = []>
hal.executable private @elementwise_transpose  {
  hal.interface @io {
    hal.interface.binding @arg0, set=0, binding=0, type="StorageBuffer"
    hal.interface.binding @arg1, set=0, binding=1, type="StorageBuffer"
    hal.interface.binding @ret0, set=0, binding=2, type="StorageBuffer"
  }
  hal.executable.variant @vulkan, target = #hal.executable.target<"spirv", "vulkan-spirv-fb"> {
    hal.executable.entry_point @elementwise_transpose attributes {
      interface = @io, ordinal = 0 : index,
      workgroup_size = [32: index, 1: index, 1: index]
    }
    builtin.module {
      func @elementwise_transpose() {
        %c0 = arith.constant 0 : index
        %arg0 = hal.interface.binding.subspan @io::@arg0[%c0] : memref<128x8xf32>
        %arg1 = hal.interface.binding.subspan @io::@arg1[%c0] : memref<128xf32>
        %ret0 = hal.interface.binding.subspan @io::@ret0[%c0] : memref<128x8xf32>
        linalg.generic {
          lowering.config = #config,
          indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                           affine_map<(d0, d1) -> (d0)>,
                           affine_map<(d0, d1) -> (d0, d1)>],
          iterator_types = ["parallel", "parallel"]
        } ins(%arg0, %arg1 : memref<128x8xf32>, memref<128xf32>)
          outs(%ret0 : memref<128x8xf32>) {
              ^bb0(%a : f32, %b : f32, %c : f32):
              %add = arith.addf %a, %b : f32
              linalg.yield %add : f32
        }
        return
      }
      hal.interface private @io  {
        hal.interface.binding @arg0, set=0, binding=0, type="StorageBuffer"
        hal.interface.binding @arg1, set=0, binding=1, type="StorageBuffer"
        hal.interface.binding @ret0, set=0, binding=2, type="StorageBuffer"
      }
    }
  }
}

// RUN: iree-opt -split-input-file -pass-pipeline='hal.executable(hal.executable.variant(builtin.module(builtin.func(iree-spirv-distribute-to-global-id))))' -canonicalize -cse %s | IreeFileCheck %s

#map0 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
hal.executable @parallel_4D attributes {sym_visibility = "private"} {
  hal.interface @io {
    hal.interface.binding @arg0, set=0, binding=0, type="StorageBuffer", access="Read"
    hal.interface.binding @arg1, set=0, binding=1, type="StorageBuffer", access="Read"
    hal.interface.binding @ret0, set=0, binding=2, type="StorageBuffer", access="Write|Discard"
  }
  hal.executable.variant @vulkan, target = #hal.executable.target<"vulkan-spirv", "vulkan-spirv-fb"> {
    hal.executable.entry_point @parallel_4D attributes {
      interface = @io,
      ordinal = 0 : index
    }
    builtin.module attributes {
      spv.target_env =
        #spv.target_env<#spv.vce<v1.3,
        [Shader], [SPV_KHR_storage_buffer_storage_class]>,
        {max_compute_workgroup_invocations = 128 : i32,
         max_compute_workgroup_size = dense<[128, 128, 64]> : vector<3xi32>}>} {
      func @parallel_4D() {
        %c0 = constant 0 : index
        %dim0 = hal.interface.load.constant offset = 0 : index
        %dim1 = hal.interface.load.constant offset = 1 : index
        %dim2 = hal.interface.load.constant offset = 2 : index
        %dim3 = hal.interface.load.constant offset = 3 : index
        %arg0 = hal.interface.binding.subspan @io::@arg0[%c0] : memref<?x?x?x?xf32>{%dim0, %dim1, %dim2, %dim3}
        %arg1 = hal.interface.binding.subspan @io::@arg1[%c0] : memref<?x?x?x?xf32>{%dim0, %dim1, %dim2, %dim3}
        %arg2 = hal.interface.binding.subspan @io::@ret0[%c0] : memref<?x?x?x?xf32>{%dim0, %dim1, %dim2, %dim3}
        linalg.generic {
           indexing_maps = [#map0, #map0, #map0],
           iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
          ins(%arg0, %arg1 : memref<?x?x?x?xf32>, memref<?x?x?x?xf32>)
         outs(%arg2 : memref<?x?x?x?xf32>) {
        ^bb0(%arg3 : f32, %arg4 : f32, %arg5 : f32):
          %0 = addf %arg3, %arg4 : f32
          linalg.yield %0 : f32
        }
        return
      }
      func private @parallel_4D__num_workgroups__
        (!shapex.ranked_shape<[?,?,?,?]>, !shapex.ranked_shape<[?,?,?,?]>,
         !shapex.ranked_shape<[?,?,?,?]>) -> (index, index, index)
      hal.interface @io attributes {sym_visibility = "private"} {
        hal.interface.binding @arg0, set=0, binding=0, type="StorageBuffer", access="Read"
        hal.interface.binding @arg1, set=0, binding=1, type="StorageBuffer", access="Read"
        hal.interface.binding @ret0, set=0, binding=2, type="StorageBuffer", access="Write|Discard"
      }
    }
  }
}
// CHECK-LABEL: func @parallel_4D
//   CHECK-DAG:     %[[C0:.+]] = constant 0 : index
//   CHECK-DAG:     %[[C1:.+]] = constant 1 : index
//   CHECK-DAG:     %[[C2:.+]] = constant 2 : index
//   CHECK-DAG:     %[[C3:.+]] = constant 3 : index
//   CHECK-DAG:     %[[UB0:.+]] = memref.dim %{{.+}}, %[[C0]]
//   CHECK-DAG:     %[[UB1:.+]] = memref.dim %{{.+}}, %[[C1]]
//   CHECK-DAG:     %[[UB2:.+]] = memref.dim %{{.+}}, %[[C2]]
//   CHECK-DAG:     %[[UB3:.+]] = memref.dim %{{.+}}, %[[C3]]
//       CHECK:     %[[T4:.+]] = muli %[[UB3]], %[[UB2]]
//       CHECK:     %[[T5:.+]] = muli %[[T4]], %[[UB1]]
//       CHECK:     %[[UB:.+]] = muli %[[T5]], %[[UB0]]
//   CHECK-DAG:     %[[BID:.+]] = "gpu.block_id"() {dimension = "x"}
//   CHECK-DAG:     %[[BDIM:.+]] = "gpu.block_dim"() {dimension = "x"}
//   CHECK-DAG:     %[[TID:.+]] = "gpu.thread_id"() {dimension = "x"}
//       CHECK:     %[[BOFFSET:.+]] = muli %[[BID]], %[[BDIM]]
//       CHECK:     %[[IV:.+]] = addi %[[BOFFSET]], %[[TID]]
//       CHECK:     %[[COND:.+]] = cmpi slt, %[[IV]], %[[UB]]
//       CHECK:     scf.if %[[COND]]
//       CHECK:       %[[IV0:.+]] = divi_signed %[[IV]], %[[T5]]
//       CHECK:       %[[T14:.+]] = remi_signed %[[IV]], %[[T5]]
//       CHECK:       %[[IV1:.+]] = divi_signed %[[T14]], %[[T4]]
//       CHECK:       %[[T16:.+]] = remi_signed %[[T14]], %[[T4]]
//       CHECK:       %[[IV2:.+]] = divi_signed %[[T16]], %[[UB3]]
//       CHECK:       %[[IV3:.+]] = remi_signed %[[T16]], %[[UB3]]
//       CHECK:       load %{{.+}}[%[[IV0]], %[[IV1]], %[[IV2]], %[[IV3]]]
//       CHECK:       load %{{.+}}[%[[IV0]], %[[IV1]], %[[IV2]], %[[IV3]]]
//       CHECK:       store %{{.+}}[%[[IV0]], %[[IV1]], %[[IV2]], %[[IV3]]]

// -----

#map0 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
hal.executable @parallel_4D_static attributes {sym_visibility = "private"} {
  hal.interface @io {
    hal.interface.binding @arg0, set=0, binding=0, type="StorageBuffer", access="Read"
    hal.interface.binding @arg1, set=0, binding=1, type="StorageBuffer", access="Read"
    hal.interface.binding @ret0, set=0, binding=2, type="StorageBuffer", access="Write|Discard"
  }
  hal.executable.variant @vulkan, target = #hal.executable.target<"vulkan-spirv", "vulkan-spirv-fb"> {
    hal.executable.entry_point @parallel_4D_static attributes {
      interface = @io,
      ordinal = 0 : index
    }
    builtin.module attributes {
      spv.target_env =
        #spv.target_env<#spv.vce<v1.3,
        [Shader], [SPV_KHR_storage_buffer_storage_class]>,
        {max_compute_workgroup_invocations = 128 : i32,
         max_compute_workgroup_size = dense<[128, 128, 64]> : vector<3xi32>}>} {
      func @parallel_4D_static() {
        %c0 = constant 0 : index
        %arg0 = hal.interface.binding.subspan @io::@arg0[%c0] : memref<3x4x5x6xf32>
        %arg1 = hal.interface.binding.subspan @io::@arg1[%c0] : memref<3x4x5x6xf32>
        %arg2 = hal.interface.binding.subspan @io::@ret0[%c0] : memref<3x4x5x6xf32>
        linalg.generic {
           indexing_maps = [#map0, #map0, #map0],
           iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
          ins(%arg0, %arg1 : memref<3x4x5x6xf32>, memref<3x4x5x6xf32>)
         outs(%arg2 : memref<3x4x5x6xf32>) {
        ^bb0(%arg3 : f32, %arg4 : f32, %arg5 : f32):
          %0 = addf %arg3, %arg4 : f32
          linalg.yield %0 : f32
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
// CHECK-LABEL: func @parallel_4D_static()
//   CHECK-DAG:     %[[C360:.+]] = constant 360 : index
//   CHECK-DAG:     %[[C120:.+]] = constant 120 : index
//   CHECK-DAG:     %[[C30:.+]] = constant 30 : index
//   CHECK-DAG:     %[[C6:.+]] = constant 6 : index
//   CHECK-DAG:     %[[BID:.+]] = "gpu.block_id"() {dimension = "x"}
//   CHECK-DAG:     %[[BDIM:.+]] = "gpu.block_dim"() {dimension = "x"}
//   CHECK-DAG:     %[[TID:.+]] = "gpu.thread_id"() {dimension = "x"}
//       CHECK:     %[[BOFFSET:.+]] = muli %[[BID]], %[[BDIM]]
//       CHECK:     %[[IV:.+]] = addi %[[BOFFSET]], %[[TID]]
//       CHECK:     %[[COND:.+]] = cmpi slt, %[[IV]], %[[C360]]
//       CHECK:     scf.if %[[COND]]
//       CHECK:       %[[IV0:.+]] = divi_signed %[[IV]], %[[C120]]
//       CHECK:       %[[T14:.+]] = remi_signed %[[IV]], %[[C120]]
//       CHECK:       %[[IV1:.+]] = divi_signed %[[T14]], %[[C30]]
//       CHECK:       %[[T16:.+]] = remi_signed %[[T14]], %[[C30]]
//       CHECK:       %[[IV2:.+]] = divi_signed %[[T16]], %[[C6]]
//       CHECK:       %[[IV3:.+]] = remi_signed %[[T16]], %[[C6]]
//       CHECK:       load %{{.+}}[%[[IV0]], %[[IV1]], %[[IV2]], %[[IV3]]]
//       CHECK:       load %{{.+}}[%[[IV0]], %[[IV1]], %[[IV2]], %[[IV3]]]
//       CHECK:       store %{{.+}}[%[[IV0]], %[[IV1]], %[[IV2]], %[[IV3]]]

// -----

#map0 = affine_map<() -> ()>
#accesses = [#map0, #map0, #map0]
#trait = {
  indexing_maps = #accesses,
  iterator_types = []
}

hal.executable @scalar_add attributes {sym_visibility = "private"} {
  hal.interface @io {
    hal.interface.binding @arg0, set=0, binding=0, type="StorageBuffer", access="Read"
    hal.interface.binding @arg1, set=0, binding=1, type="StorageBuffer", access="Read"
    hal.interface.binding @ret0, set=0, binding=2, type="StorageBuffer", access="Write|Discard"
  }
  hal.executable.variant @vulkan, target = #hal.executable.target<"vulkan-spirv", "vulkan-spirv-fb"> {
    hal.executable.entry_point @scalar_add attributes {
      interface = @io,
      ordinal = 0 : index
    }
    builtin.module attributes {
      spv.target_env =
        #spv.target_env<#spv.vce<v1.3,
        [Shader], [SPV_KHR_storage_buffer_storage_class]>,
        {max_compute_workgroup_invocations = 128 : i32,
         max_compute_workgroup_size = dense<[128, 128, 64]> : vector<3xi32>}>} {
      func @scalar_add() attributes {hal.num_workgroups_fn = @scalar_add__num_workgroups__} {
        %c0 = constant 0 : index
        %arg0 = hal.interface.binding.subspan @io::@arg0[%c0] : memref<f32>
        %arg1 = hal.interface.binding.subspan @io::@arg1[%c0] : memref<f32>
        %arg2 = hal.interface.binding.subspan @io::@ret0[%c0] : memref<f32>
        linalg.generic #trait
          ins(%arg0, %arg1 : memref<f32>, memref<f32>)
         outs(%arg2 : memref<f32>) {
        ^bb0(%arg3 : f32, %arg4 : f32, %arg5 : f32):
          %0 = addf %arg3, %arg4 : f32
          linalg.yield %0 : f32
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
// CHECK-LABEL: func @scalar_add()
//       CHECK:     load
//  CHECK-NEXT:     load
//  CHECK-NEXT:     addf
//  CHECK-NEXT:     store
//  CHECK-NEXT:     return

// -----

// TODO(GH-4901): Convert these tests back to use dynamic shapes when linalg on tensors becomes default.
hal.executable @reduce_sum attributes {sym_visibility = "private"} {
  hal.interface @io {
    hal.interface.binding @arg0, set=0, binding=0, type="StorageBuffer", access="Read"
    hal.interface.binding @arg1, set=0, binding=1, type="StorageBuffer", access="Read"
    hal.interface.binding @ret0, set=0, binding=2, type="StorageBuffer", access="Write|Discard"
  }
  hal.executable.variant @vulkan, target = #hal.executable.target<"vulkan-spirv", "vulkan-spirv-fb"> {
    hal.executable.entry_point @reduce_sum attributes {
      interface = @io,
      ordinal = 0 : index
    }
    builtin.module {
      func @reduce_sum() {
        %c0 = constant 0 : index
        %arg0 = hal.interface.binding.subspan @io::@arg0[%c0] : memref<40x50x75xf32>
        %arg1 = hal.interface.binding.subspan @io::@arg1[%c0] : memref<f32>
        %arg2 = hal.interface.binding.subspan @io::@ret0[%c0] : memref<40xf32>
        linalg.generic {
          indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>,
                           affine_map<(d0, d1, d2) -> ()>,
                           affine_map<(d0, d1, d2) -> (d0)>],
          iterator_types = ["parallel", "reduction", "reduction"]}
          ins(%arg0, %arg1 : memref<40x50x75xf32>, memref<f32>)
          outs(%arg2 : memref<40xf32>) {
        ^bb0(%arg6: f32, %arg7: f32, %arg8: f32):   // no predecessors
          %idx1 = linalg.index 1 : index
          %idx2 = linalg.index 2 : index
          %zero = constant 0 : index
          %0 = cmpi eq, %idx2, %zero : index
          %1 = cmpi eq, %idx1, %zero : index
          %2 = and %0, %1 : i1
          %3 = select %2, %arg7, %arg8 : f32
          %4 = addf %arg6, %3 : f32
          linalg.yield %4 : f32
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
//CHECK-LABEL: func @reduce_sum
//   CHECK-DAG:     %[[C0:.+]] = constant 0 : index
//   CHECK-DAG:     %[[C40:.+]] = constant 40 : index
//   CHECK-DAG:     %[[C50:.+]] = constant 50 : index
//   CHECK-DAG:     %[[C75:.+]] = constant 75 : index
//       CHECK:     %[[COND:.+]] = cmpi slt, %{{.+}}, %[[C40]]
//       CHECK:     scf.if %[[COND]]
//       CHECK:       scf.for %[[IV0:.+]] = %{{.+}} to %[[C50]]
//       CHECK:         scf.for %[[IV1:.+]] = %{{.+}} to %[[C75]]
//   CHECK-DAG:           %[[ISZERO0:.+]] = cmpi eq, %[[IV0]], %[[C0]]
//   CHECK-DAG:           %[[ISZERO1:.+]] = cmpi eq, %[[IV1]], %[[C0]]

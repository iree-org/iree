// RUN: iree-opt -split-input-file -iree-hal-transformation-pipeline -iree-hal-target-backends=interpreter-bytecode %s | IreeFileCheck %s -check-prefix=INTERP
// RUN: iree-opt -split-input-file -iree-hal-transformation-pipeline -iree-hal-target-backends=vulkan-spirv %s | IreeFileCheck %s -check-prefix=VKSPV

flow.executable @simpleMath_ex_dispatch_0 {
  flow.dispatch.entry @simpleMath_rgn_dispatch_0 attributes {
      workload = dense<[4, 1, 1]> : vector<3xi32>
  }
  module {
    func @simpleMath_rgn_dispatch_0(%arg0: tensor<4xf32>) -> tensor<4xf32> {
      %0 = xla_hlo.add %arg0, %arg0 : tensor<4xf32>
      return %0 : tensor<4xf32>
    }
  }
}

// INTERP-LABEL: hal.executable @simpleMath_ex_dispatch_0 {
// INTERP-NEXT:   hal.interface @legacy_io {
// INTERP-DAG:      hal.interface.binding @arg0, set=0, binding=0, type="StorageBuffer", access="Read"
// INTERP-DAG:      hal.interface.binding @ret0, set=0, binding=1, type="StorageBuffer", access="Write|Discard"
// INTERP-NEXT:   }
// INTERP-NEXT:   hal.executable.entry_point @simpleMath_rgn_dispatch_0 attributes {interface = @legacy_io, ordinal = 0 : i32, signature = (tensor<4xf32>) -> tensor<4xf32>, workgroup_size = dense<1> : vector<3xi32>
// INTERP-NEXT:   hal.executable.binary attributes {
// INTERP-SAME:     data = dense
// INTERP-SAME:     format = 1230128453 : i32} {
// INTERP-NEXT:     module {
// INTERP-NEXT:       func @simpleMath_rgn_dispatch_0(%arg0: memref<4xf32>, %arg1: memref<4xf32>) attributes {
// INTERP-SAME:           iree.executable.export,
// INTERP-SAME:           iree.ordinal = 0 : i32} {
// INTERP-NEXT:         %0 = "iree_ll_interp.alloc_heap"() : () -> memref<4xf32>
// INTERP-NEXT:         "iree_ll_interp.add_f"(%arg0, %arg0, %0) : (memref<4xf32>, memref<4xf32>, memref<4xf32>) -> ()
// INTERP-NEXT:         %1 = "iree_ll_interp.constant"() {value = dense<0> : tensor<1xi64>} : () -> memref<1xi64>
// INTERP-NEXT:         %2 = "iree_ll_interp.constant"() {value = dense<4> : tensor<1xi64>} : () -> memref<1xi64>
// INTERP-NEXT:         "iree_ll_interp.dynamic_copy"(%0, %1, %arg1, %1, %2) : (memref<4xf32>, memref<1xi64>, memref<4xf32>, memref<1xi64>, memref<1xi64>) -> ()
// INTERP-NEXT:         return
// INTERP-NEXT:       }
// INTERP-NEXT:     }
// INTERP-NEXT:   }
// INTERP-NEXT: }

// VKSPV-LABEL: hal.executable @simpleMath_ex_dispatch_0 {
// VKSPV-NEXT:   hal.interface @legacy_io {
// VKSPV-DAG:      hal.interface.binding @arg0, set=0, binding=0, type="StorageBuffer", access="Read"
// VKSPV-DAG:      hal.interface.binding @ret0, set=0, binding=1, type="StorageBuffer", access="Write|Discard"
// VKSPV-NEXT:   }
// VKSPV-NEXT:   hal.executable.entry_point @simpleMath_rgn_dispatch_0 attributes {interface = @legacy_io, ordinal = 0 : i32, signature = (tensor<4xf32>) -> tensor<4xf32>, workgroup_size = dense<[32, 1, 1]> : vector<3xi32>
// VKSPV-NEXT:   hal.executable.binary attributes {
// VKSPV-SAME:     data = dense
// VKSPV-SAME:     format = 1397773893 : i32} {
// VKSPV-NEXT:     module {
// VKSPV-NEXT:       spv.module "Logical" "GLSL450" {
//  VKSPV-DAG:         spv.globalVariable [[GLOBALID:@.*]] built_in("GlobalInvocationId")
//  VKSPV-DAG:         spv.globalVariable [[NUMWORKGROUPS:@.*]] built_in("NumWorkgroups")
//      VKSPV:         spv.EntryPoint "GLCompute" @simpleMath_rgn_dispatch_0, [[GLOBALID]], [[NUMWORKGROUPS]]
// VKSPV-NEXT:         spv.ExecutionMode @simpleMath_rgn_dispatch_0 "LocalSize", 32, 1, 1
// VKSPV-NEXT:       } attributes {
// VKSPV-SAME:         capabilities = ["Shader"],
// VKSPV-SAME:         extensions = ["SPV_KHR_storage_buffer_storage_class"]}
// VKSPV-NEXT:     }
// VKSPV-NEXT:   }
// VKSPV-NEXT: }

// -----

flow.executable @reduction_ex_reduce_0_dim_0 {
  flow.reduction.entry @reduction_rgn_reduce_0_dim_0_entry apply(@reduction_rgn_reduce_0_dim_0) attributes {
    dimension = 1 : i32,
    workgroup_size = dense<[32, 1, 1]> : vector<3xi32>,
    workload = dense<[4, 1, 1]> : vector<3xi32>
  }
  module {
    func @reduction_rgn_reduce_0_dim_0_entry(tensor<4x8xf32>, tensor<f32>) -> tensor<4xf32>
    func @reduction_rgn_reduce_0_dim_0(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
      %0 = xla_hlo.add %arg0, %arg1 : tensor<f32>
      return %0 : tensor<f32>
    }
  }
}

// INTERP-LABEL: hal.executable @reduction_ex_reduce_0_dim_0 {
// INTERP-NEXT:   hal.interface @legacy_io {
// INTERP-DAG:      hal.interface.binding @arg0, set=0, binding=0, type="StorageBuffer", access="Read"
// INTERP-DAG:      hal.interface.binding @arg1, set=0, binding=1, type="StorageBuffer", access="Read"
// INTERP-DAG:      hal.interface.binding @ret0, set=0, binding=2, type="StorageBuffer", access="Write|Discard"
// INTERP-NEXT:   }
// INTERP-NEXT:   hal.executable.entry_point @reduction_rgn_reduce_0_dim_0_entry attributes {
// INTERP-SAME:     interface = @legacy_io,
// INTERP-SAME:     ordinal = 0 : i32,
// INTERP-SAME:     signature = (tensor<4x8xf32>, tensor<f32>) -> tensor<4xf32>,
// INTERP-SAME:     workgroup_size = dense<1> : vector<3xi32>
// INTERP-SAME:   }
// INTERP-NEXT:   hal.executable.binary attributes {
// INTERP-SAME:     data = dense
// INTERP-SAME:     format = 1230128453 : i32} {
// INTERP-NEXT:     module {
// INTERP-NEXT:       func @reduction_rgn_reduce_0_dim_0_entry(%arg0: memref<4x8xf32>, %arg1: memref<f32>, %arg2: memref<4xf32>) attributes {
// INTERP-SAME:           iree.executable.export,
// INTERP-SAME:           iree.executable.reduction,
// INTERP-SAME:           iree.ordinal = 0 : i32} {
//      INTERP:         "iree_ll_interp.reduce_sum_f"(%arg0, %arg1, %0) {dimension = 1 : i32} : (memref<4x8xf32>, memref<f32>, memref<4xf32>) -> ()

// VKSPV-LABEL: hal.executable @reduction_ex_reduce_0_dim_0 {
// VKSPV-NEXT:   hal.interface @legacy_io {
// VKSPV-DAG:      hal.interface.binding @arg0, set=0, binding=0, type="StorageBuffer", access="Read"
// VKSPV-DAG:      hal.interface.binding @arg1, set=0, binding=1, type="StorageBuffer", access="Read"
// VKSPV-DAG:      hal.interface.binding @ret0, set=0, binding=2, type="StorageBuffer", access="Write|Discard"
// VKSPV-NEXT:   }
// VKSPV-NEXT:   hal.executable.entry_point @reduction_rgn_reduce_0_dim_0_entry attributes {
// VKSPV-SAME:     interface = @legacy_io,
// VKSPV-SAME:     ordinal = 0 : i32,
// VKSPV-SAME:     signature = (tensor<4x8xf32>, tensor<f32>) -> tensor<4xf32>,
// VKSPV-SAME:     workgroup_size = dense<1> : vector<3xi32>
// VKSPV-SAME:   }
// VKSPV-NEXT:   hal.executable.binary attributes {
// VKSPV-SAME:     data = dense
// VKSPV-SAME:     format = 1397773893 : i32} {

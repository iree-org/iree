// RUN: iree-opt -split-input-file -iree-hal-transformation-pipeline -iree-hal-target-backends=vulkan-spirv %s | IreeFileCheck %s -check-prefix=VKSPV

flow.executable @simpleMath_ex_dispatch_0 {
  flow.dispatch.entry @simpleMath_rgn_dispatch_0 attributes {
      workload = 4 : index
  }
  module {
    func @simpleMath_rgn_dispatch_0(%arg0: tensor<4xf32>) -> tensor<4xf32> {
      %0 = mhlo.add %arg0, %arg0 : tensor<4xf32>
      return %0 : tensor<4xf32>
    }
  }
}

// VKSPV-LABEL: hal.executable @simpleMath_ex_dispatch_0
// VKSPV-NEXT:   hal.interface @legacy_io {
// VKSPV-DAG:      hal.interface.binding @arg0, set=0, binding=0, type="StorageBuffer", access="Read"
// VKSPV-DAG:      hal.interface.binding @ret0, set=0, binding=1, type="StorageBuffer", access="Write|Discard"
// VKSPV-NEXT:   }
// VKSPV-NEXT:   hal.executable.entry_point @simpleMath_rgn_dispatch_0 attributes {interface = @legacy_io, ordinal = 0 : i32, signature = (tensor<4xf32>) -> tensor<4xf32>
// VKSPV-NEXT:   hal.executable.binary attributes {
// VKSPV-SAME:     data = dense
// VKSPV-SAME:     format = 1397773893 : i32}

// -----

flow.executable @reduction_ex_reduce_0_dim_0 {
  flow.dispatch.entry @reduction_rgn_reduce_0_dim_0_entry attributes {
    workload = 4 : index
  }
  module {
    func @reduction_rgn_reduce_0_dim_0_entry(%arg0 : tensor<4x8xf32>, %arg1 : tensor<f32>) -> tensor<4xf32> {
      %0 = "mhlo.reduce"(%arg0, %arg1) ( {
      ^bb0(%arg3: tensor<f32>, %arg4: tensor<f32>):
        %1 = mhlo.add %arg3, %arg4 : tensor<f32>
        "mhlo.return"(%1) : (tensor<f32>) -> ()
      }) {dimensions = dense<1> : tensor<1xi64>} : (tensor<4x8xf32>, tensor<f32>) -> tensor<4xf32>
      return %0 : tensor<4xf32>
    }
  }
}

// VKSPV-LABEL: hal.executable @reduction_ex_reduce_0_dim_0
// VKSPV-NEXT:   hal.interface @legacy_io {
// VKSPV-DAG:      hal.interface.binding @arg0, set=0, binding=0, type="StorageBuffer", access="Read"
// VKSPV-DAG:      hal.interface.binding @arg1, set=0, binding=1, type="StorageBuffer", access="Read"
// VKSPV-DAG:      hal.interface.binding @ret0, set=0, binding=2, type="StorageBuffer", access="Write|Discard"
// VKSPV-NEXT:   }
// VKSPV-NEXT:   hal.executable.entry_point @reduction_rgn_reduce_0_dim_0_entry attributes {
// VKSPV-SAME:     interface = @legacy_io,
// VKSPV-SAME:     ordinal = 0 : i32,
// VKSPV-SAME:     signature = (tensor<4x8xf32>, tensor<f32>) -> tensor<4xf32>
// VKSPV-SAME:   }
// VKSPV-NEXT:   hal.executable.binary attributes {
// VKSPV-SAME:     data = dense
// VKSPV-SAME:     format = 1397773893 : i32} {

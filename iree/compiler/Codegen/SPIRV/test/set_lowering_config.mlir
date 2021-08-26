// RUN: iree-opt -split-input-file -pass-pipeline='hal.executable(hal.executable.variant(iree-spirv-lower-executable-target-pass{test-lowering-configuration=true}))' %s | IreeFileCheck %s

hal.executable @static_1d_sort attributes {sym_visibility = "private"} {
  hal.interface @io {
    hal.interface.binding @s0b0_rw_external, set=0, binding=0, type="StorageBuffer", access="Read|Write"
  }
  hal.executable.variant @vulkan_spirv_fb, target = #hal.executable.target<"vulkan-spirv", "vulkan-spirv-fb"> {
    hal.executable.entry_point @static_1d_sort attributes {interface = @io, ordinal = 0 : index}
    builtin.module attributes {
      spv.target_env = #spv.target_env<#spv.vce<v1.4, [Shader], []>, ARM:IntegratedGPU, {
        max_compute_shared_memory_size = 32768 : i32,
        max_compute_workgroup_invocations = 512 : i32,
        max_compute_workgroup_size = dense<512> : vector<3xi32>,
        subgroup_size = 16 : i32}>
    } {
      builtin.func @static_1d_sort() {
        %c0 = constant 0 : index
        %0 = hal.interface.binding.subspan @io::@s0b0_rw_external[%c0] : !flow.dispatch.tensor<readwrite:4xi32>
        %1 = flow.dispatch.tensor.load %0, offsets = [], sizes = [], strides = [] : !flow.dispatch.tensor<readwrite:4xi32> -> tensor<4xi32>
        %2 = linalg_ext.sort dimension(0) {__internal_linalg_transform__ = "workgroup"} outs(%1 : tensor<4xi32>)  {
        ^bb0(%arg0: i32, %arg1: i32):  // no predecessors
          %3 = cmpi slt, %arg0, %arg1 : i32
          linalg_ext.yield %3 : i1
        } -> tensor<4xi32>
        flow.dispatch.tensor.store %2, %0, offsets = [], sizes = [], strides = [] : tensor<4xi32> -> !flow.dispatch.tensor<readwrite:4xi32>
        return
      }
      hal.interface @io attributes {sym_visibility = "private"} {
        hal.interface.binding @s0b0_rw_external, set=0, binding=0, type="StorageBuffer", access="Read|Write"
      }
    }
  }
}

// Check that the workgroup size is (1, 1, 1) for serializing the computation.

// CHECK-LABEL: hal.executable.entry_point @static_1d_sort
//  CHECK-SAME:   translation.info = {passPipeline = 6 : i32}
//  CHECK-SAME:   workgroup_size = [1 : index, 1 : index, 1 : index]

// -----

hal.executable @static_2d_fft attributes {sym_visibility = "private"} {
  hal.interface @io {
    hal.interface.binding @s0b0_rw_external, set=0, binding=0, type="StorageBuffer", access="Read|Write"
    hal.interface.binding @s0b1_rw_external, set=0, binding=1, type="StorageBuffer", access="Read|Write"
  }
  hal.executable.variant @vulkan_spirv_fb, target = #hal.executable.target<"vulkan-spirv", "vulkan-spirv-fb"> {
    hal.executable.entry_point @static_2d_fft attributes {interface = @io, ordinal = 0 : index}
    builtin.module attributes {
      spv.target_env = #spv.target_env<#spv.vce<v1.4, [Shader], []>, ARM:IntegratedGPU, {
        max_compute_shared_memory_size = 32768 : i32,
        max_compute_workgroup_invocations = 512 : i32,
        max_compute_workgroup_size = dense<512> : vector<3xi32>,
        subgroup_size = 16 : i32}>
    } {
      builtin.func @static_2d_fft() {
        %cst = constant dense<[-0.000000e+00, -1.000000e+00]> : tensor<2xf32>
        %cst_0 = constant dense<[1.000000e+00, 6.12323426E-17]> : tensor<2xf32>
        %c2 = constant 2 : index
        %c0 = constant 0 : index
        %0 = memref.buffer_cast %cst_0 : memref<2xf32>
        %1 = memref.buffer_cast %cst : memref<2xf32>
        %2 = hal.interface.binding.subspan @io::@s0b0_rw_external[%c0] : memref<1x32xf32>
        %3 = hal.interface.binding.subspan @io::@s0b1_rw_external[%c0] : memref<1x32xf32>
        linalg_ext.fft {__internal_linalg_transform__ = "workgroup"} ins(%c2, %0, %1 : index, memref<2xf32>, memref<2xf32>) outs(%2, %3 : memref<1x32xf32>, memref<1x32xf32>)
        return
      }
      hal.interface @io attributes {sym_visibility = "private"} {
        hal.interface.binding @s0b0_rw_external, set=0, binding=0, type="StorageBuffer", access="Read|Write"
        hal.interface.binding @s0b1_rw_external, set=0, binding=1, type="StorageBuffer", access="Read|Write"
      }
    }
  }
}

// Check that the workgroup size is (1, 1, 1) for serializing the computation.

// CHECK-LABEL: hal.executable.entry_point @static_2d_fft
//  CHECK-SAME:   translation.info = {passPipeline = 6 : i32}
//  CHECK-SAME:   workgroup_size = [1 : index, 1 : index, 1 : index]

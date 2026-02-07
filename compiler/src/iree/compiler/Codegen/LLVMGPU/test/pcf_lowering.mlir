// RUN: iree-opt %s --pass-pipeline="builtin.module(func.func(iree-pcf-resolve-tokens, iree-pcf-convert-sref-to-memref, iree-pcf-lower-structural-pcf))" --split-input-file | FileCheck %s

func.func @pcf_workgroup_loop(%arg0: memref<64xf32>) {
  %c64 = arith.constant 64 : index
  pcf.loop scope(#iree_codegen.workgroup_scope) count(%c64)
    execute[%iv: index] {
    %c0 = arith.constant 0.0 : f32
    memref.store %c0, %arg0[%iv] : memref<64xf32>
    pcf.return
  }
  return
}

// CHECK-LABEL: @pcf_workgroup_loop
// Verify workgroup IDs are materialized.
//       CHECK:   hal.interface.workgroup.id
//       CHECK:   hal.interface.workgroup.count

// -----

func.func @pcf_subgroup_generic() {
  pcf.generic scope(#iree_gpu.subgroup_scope)
    execute[%id: index, %n: index] {
    pcf.return
  }
  return
}

// CHECK-LABEL: @pcf_subgroup_generic
// Verify subgroup IDs are materialized.
//       CHECK:   gpu.subgroup_id
//       CHECK:   gpu.num_subgroups

// -----

func.func @pcf_lane_generic() {
  pcf.generic scope(#iree_gpu.lane_scope)
    execute[%id: index, %n: index] {
    pcf.return
  }
  return
}

// CHECK-LABEL: @pcf_lane_generic
// Verify lane IDs are materialized.
//       CHECK:   gpu.lane_id
//       CHECK:   gpu.subgroup_size

// -----

// Test nested workgroup -> subgroup -> lane pattern.
func.func @pcf_nested_gpu_scopes(%arg0: memref<64xf32>) {
  %c64 = arith.constant 64 : index
  pcf.loop scope(#iree_codegen.workgroup_scope) count(%c64)
    execute[%wg_id: index] {
    pcf.generic scope(#iree_gpu.subgroup_scope)
      execute[%sg_id: index, %sg_n: index] {
      pcf.generic scope(#iree_gpu.lane_scope)
        execute[%lane_id: index, %lane_n: index] {
        %c0 = arith.constant 0.0 : f32
        memref.store %c0, %arg0[%lane_id] : memref<64xf32>
        pcf.return
      }
      pcf.return
    }
    pcf.return
  }
  return
}

// CHECK-LABEL: @pcf_nested_gpu_scopes
// Verify workgroup, subgroup, and lane IDs are materialized.
//       CHECK:   hal.interface.workgroup.id
//       CHECK:   gpu.subgroup_id
//       CHECK:   gpu.lane_id

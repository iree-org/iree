// RUN: iree-opt %s --split-input-file --mlir-print-local-scope \
// RUN:   --pass-pipeline="builtin.module(hal.executable(hal.executable.variant(builtin.module(func.func(iree-codegen-gpu-canonicalize-ids, canonicalize, cse)))))" | \
// RUN:   FileCheck %s

// Tests for the case where workgroup/subgroup size comes from
// hal.executable.export attributes rather than translation_info.

#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>
]>

// Test: gpu.subgroup_size and gpu.block_dim fold from export attributes.
hal.executable private @test_fold_sizes {
  hal.executable.variant public @variant target(<"rocm", "rocm-hsaco-fb">) {
    hal.executable.export public @fold_sizes ordinal(0) layout(#pipeline_layout)
        attributes {workgroup_size = [128 : index, 2 : index, 1 : index],
                    subgroup_size = 64 : index}
    builtin.module {
      func.func @fold_sizes() -> (index, index, index, index) {
        %sg = gpu.subgroup_size : index
        %bx = gpu.block_dim x
        %by = gpu.block_dim y
        %bz = gpu.block_dim z
        return %sg, %bx, %by, %bz : index, index, index, index
      }
    }
  }
}

// CHECK-LABEL: func.func @fold_sizes
//   CHECK-DAG:   %[[C64:.+]] = arith.constant 64 : index
//   CHECK-DAG:   %[[C128:.+]] = arith.constant 128 : index
//   CHECK-DAG:   %[[C2:.+]] = arith.constant 2 : index
//   CHECK-DAG:   %[[C1:.+]] = arith.constant 1 : index
//       CHECK:   return %[[C64]], %[[C128]], %[[C2]], %[[C1]]

// -----

#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>
]>

// Test: gpu.subgroup_id rewrites using export attributes.
hal.executable private @test_rewrite_subgroup_id {
  hal.executable.variant public @variant target(<"rocm", "rocm-hsaco-fb">) {
    hal.executable.export public @rewrite_subgroup_id ordinal(0) layout(#pipeline_layout)
        attributes {workgroup_size = [128 : index, 2 : index, 1 : index],
                    subgroup_size = 64 : index}
    builtin.module {
      func.func @rewrite_subgroup_id() -> index {
        %0 = gpu.subgroup_id : index
        return %0 : index
      }
    }
  }
}

// The linearization is: tid.x + 128 * tid.y (z-dim is 1, folded away).
// Then divui by 64.
// CHECK-LABEL: func.func @rewrite_subgroup_id
//   CHECK-DAG:   %[[C64:.+]] = arith.constant 64 : index
//   CHECK-DAG:   %[[C128:.+]] = arith.constant 128 : index
//   CHECK-DAG:   %[[TX:.+]] = gpu.thread_id x
//   CHECK-DAG:   %[[TY:.+]] = gpu.thread_id y
//       CHECK:   %[[MUL:.+]] = arith.muli %[[TY]], %[[C128]]
//       CHECK:   %[[LINEAR:.+]] = arith.addi %[[TX]], %[[MUL]]
//       CHECK:   %[[RESULT:.+]] = arith.divui %[[LINEAR]], %[[C64]]
//       CHECK:   return %[[RESULT]]

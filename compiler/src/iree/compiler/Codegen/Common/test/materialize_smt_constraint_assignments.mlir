// RUN: iree-opt --split-input-file --verify-diagnostics --iree-codegen-test-materialize-smt-constraint-assignments %s | FileCheck %s

// Materializes a flat knobs dictionary into GPU lowering/translation attrs.
iree_codegen.smt.constraints
    target = <set = 0>,
    pipeline = #iree_gpu.pipeline<VectorDistribute>,
    knobs = {
      workgroup = [#iree_codegen.smt.int_knob<"wg_0">, #iree_codegen.smt.int_knob<"wg_1">, 0],
      reduction = [0, 0, #iree_codegen.smt.int_knob<"red_2">],
      mma_kind = #iree_codegen.smt.one_of_knob<"mma_idx", [#iree_gpu.mma_layout<MFMA_F32_16x16x16_F16>, #iree_gpu.mma_layout<MFMA_F32_32x32x8_F16>]>,
      subgroup_basis = [[#iree_codegen.smt.int_knob<"sg_m_cnt">, #iree_codegen.smt.int_knob<"sg_n_cnt">, 1], [0, 1, 2]],
      workgroup_size = [#iree_codegen.smt.int_knob<"wg_size_x">, #iree_codegen.smt.int_knob<"wg_size_y">, #iree_codegen.smt.int_knob<"wg_size_z">],
      subgroup_size = #iree_codegen.smt.int_knob<"sg_size">
    }
    dims() attributes {
      test.assignments = {
        mma_idx = 1 : i64,
        red_2 = 8 : i64,
        sg_m_cnt = 2 : i64,
        sg_n_cnt = 1 : i64,
        sg_size = 64 : i64,
        unused_assignment = 42 : i64,
        wg_0 = 64 : i64,
        wg_1 = 128 : i64,
        wg_size_x = 128 : i64,
        wg_size_y = 1 : i64,
        wg_size_z = 1 : i64
      }
    } {
    }

// CHECK: #translation = #iree_codegen.translation_info<
// CHECK-SAME: pipeline = #iree_gpu.pipeline<VectorDistribute>
// CHECK-SAME: workgroup_size = [128, 1, 1]
// CHECK-SAME: subgroup_size = 64>
// CHECK: #compilation = #iree_codegen.compilation_info<
// CHECK-SAME: lowering_config = #iree_gpu.lowering_config<
// CHECK-SAME: mma_kind = #iree_gpu.mma_layout<MFMA_F32_32x32x8_F16>
// CHECK-SAME: reduction = [0, 0, 8]
// CHECK-SAME: subgroup_basis = {{\[}}[2, 1, 1], [0, 1, 2]]
// CHECK-SAME: workgroup = [64, 128, 0]
// CHECK-SAME: translation_info = #translation>
// CHECK:       iree_codegen.smt.constraints
// CHECK:       dims() attributes {test.materialized_compilation_info = #compilation}

// -----

// Materializes explicitly nested lowering_config and translation_info dicts.
iree_codegen.smt.constraints
    target = <set = 0>,
    pipeline = #iree_gpu.pipeline<VectorDistribute>,
    knobs = {
      lowering_config = {
        extra = 7,
        workgroup = [#iree_codegen.smt.int_knob<"nwg_0">, #iree_codegen.smt.int_knob<"nwg_1">, 0]
      },
      translation_info = {
        prefetch = #iree_codegen.smt.int_knob<"prefetch">,
        workgroup_size = [#iree_codegen.smt.int_knob<"nwg_size_x">, #iree_codegen.smt.int_knob<"nwg_size_y">, #iree_codegen.smt.int_knob<"nwg_size_z">],
        subgroup_size = #iree_codegen.smt.int_knob<"nsg_size">
      }
    }
    dims() attributes {
      test.assignments = {
        nsg_size = 32 : i64,
        nwg_0 = 32 : i64,
        nwg_1 = 64 : i64,
        nwg_size_x = 64 : i64,
        nwg_size_y = 1 : i64,
        nwg_size_z = 1 : i64,
        prefetch = 9 : i64
      }
    } {
    }

// CHECK:       #translation = #iree_codegen.translation_info<
// CHECK-SAME:  pipeline = #iree_gpu.pipeline<VectorDistribute>
// CHECK-SAME:  workgroup_size = [64, 1, 1]
// CHECK-SAME:  subgroup_size = 32, {prefetch = 9 : i64}>
// CHECK:       #compilation = #iree_codegen.compilation_info<
// CHECK-SAME:  lowering_config = #iree_gpu.lowering_config<
// CHECK-SAME:  extra = 7 : i64
// CHECK-SAME:  workgroup = [32, 64, 0]
// CHECK-SAME:  translation_info = #translation>
// CHECK:       iree_codegen.smt.constraints
// CHECK:       dims() attributes {test.materialized_compilation_info = #compilation}

// -----

// Materializes defaults when workgroup_size/subgroup_size are omitted.
iree_codegen.smt.constraints
    target = <set = 0>,
    pipeline = #iree_gpu.pipeline<VectorDistribute>,
    knobs = {
      workgroup = [#iree_codegen.smt.int_knob<"wg_0">, #iree_codegen.smt.int_knob<"wg_1">, 0]
    }
    dims() attributes {
      test.assignments = {
        wg_0 = 64 : i64,
        wg_1 = 128 : i64
      }
    } {
    }

// CHECK:       #translation = #iree_codegen.translation_info<pipeline = #iree_gpu.pipeline<VectorDistribute>>
// CHECK:       #compilation = #iree_codegen.compilation_info<lowering_config = #iree_gpu.lowering_config<{workgroup = [64, 128, 0]}>, translation_info = #translation>
// CHECK:       iree_codegen.smt.constraints
// CHECK:       dims() attributes {test.materialized_compilation_info = #compilation}

// -----

// Reports a missing knob assignment.
// expected-error @below {{failed to materialize compilation_info from constraints: missing assignment for knob 'mma_idx'}}
iree_codegen.smt.constraints
    target = <set = 0>,
    pipeline = #iree_gpu.pipeline<VectorDistribute>,
    knobs = {
      mma_kind = #iree_codegen.smt.one_of_knob<"mma_idx", [#iree_gpu.mma_layout<MFMA_F32_16x16x16_F16>, #iree_gpu.mma_layout<MFMA_F32_32x32x8_F16>]>
    }
    dims() {
    }

// -----

// Reports an out-of-range one_of knob assignment.
// expected-error @below {{failed to materialize compilation_info from constraints: assignment for knob 'mma_idx' is out of range: 99 is not in [0, 2)}}
iree_codegen.smt.constraints
    target = <set = 0>,
    pipeline = #iree_gpu.pipeline<VectorDistribute>,
    knobs = {
      mma_kind = #iree_codegen.smt.one_of_knob<"mma_idx", [#iree_gpu.mma_layout<MFMA_F32_16x16x16_F16>, #iree_gpu.mma_layout<MFMA_F32_32x32x8_F16>]>
    }
    dims() attributes {
      test.assignments = {mma_idx = 99 : i64}
    } {
    }

// -----

// Reports non-integer values in the test assignment dictionary.
// expected-error @below {{expected test assignment 'wg_0' to be an integer}}
iree_codegen.smt.constraints
    target = <set = 0>,
    pipeline = #iree_gpu.pipeline<VectorDistribute>,
    knobs = {
      workgroup = [#iree_codegen.smt.int_knob<"wg_0">, 1, 1]
    }
    dims() attributes {
      test.assignments = {wg_0 = "one"}
    } {
    }

// -----

// Reports missing named materialization outputs generically.
// expected-error @below {{failed to materialize compilation_info from constraints: pipeline does not support materializing configuration attr 'compilation_info'}}
iree_codegen.smt.constraints
    target = <set = 0>,
    pipeline = #iree_codegen.transform_dialect_codegen,
    knobs = {
      workgroup_size = [1, 1, 1],
      subgroup_size = 64
    }
    dims() {
    }

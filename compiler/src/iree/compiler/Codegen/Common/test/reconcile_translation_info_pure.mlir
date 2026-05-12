// RUN: iree-opt --split-input-file --pass-pipeline="builtin.module(iree-codegen-reconcile-translation-info)" %s | FileCheck %s

// Different from other files, this file is purely testing the
// `ReconcileTranslationInfoPass`. Ideally, each test file should test each pass
// individually, but the current setup does not allow it.

// Tests for the fallback stub `iree_codegen.dispatch_config` op that
// `ReconcileTranslationInfoPass` creates when no `dispatch_config` exists for
// an entry-point function (e.g. iree-opt pipeline tests that bypass
// `CreateDispatchConfigPass`).

// CHECK-LABEL: func.func @no_workload_ordinals
//       CHECK: iree_codegen.dispatch_config @no_workload_ordinals workgroup_size = [64, 1, 1]
//  CHECK-NEXT:   %[[X:.+]], %[[Y:.+]], %[[Z:.+]] = iree_tensor_ext.dispatch.workgroup_count_from_slice()
//  CHECK-NEXT:   iree_codegen.yield %[[X]], %[[Y]], %[[Z]]
func.func @no_workload_ordinals() attributes {
  translation_info = #iree_codegen.translation_info<pipeline = #iree_codegen.no_pipeline workgroup_size = [64]>
} {
  return
}

// -----

// When the function uses workload ordinals, the stub block argument list is
// sized to cover every referenced ordinal (max ordinal + 1), so that the
// later `ResolveWorkgroupCountHintsPass` can map each ordinal to a stub
// workload value.

#pipeline_layout = #hal.pipeline.layout<constants = 4, bindings = []>

// CHECK-LABEL: func.func @with_workload_ordinals
//       CHECK: iree_codegen.dispatch_config @with_workload_ordinals workgroup_size = [128, 1, 1]
//  CHECK-NEXT:   ^bb0(%[[A0:.+]]: index, %[[A1:.+]]: index, %[[A2:.+]]: index, %[[A3:.+]]: index):
//  CHECK-NEXT:   %[[X:.+]], %[[Y:.+]], %[[Z:.+]] = iree_tensor_ext.dispatch.workgroup_count_from_slice(%[[A0]], %[[A1]], %[[A2]], %[[A3]])
//  CHECK-NEXT:   iree_codegen.yield %[[X]], %[[Y]], %[[Z]]
func.func @with_workload_ordinals() attributes {
  translation_info = #iree_codegen.translation_info<pipeline = #iree_codegen.no_pipeline workgroup_size = [128]>
} {
  %p0 = hal.interface.constant.load layout(#pipeline_layout) ordinal(0) : i32
  %p1 = hal.interface.constant.load layout(#pipeline_layout) ordinal(1) : i32
  %p2 = hal.interface.constant.load layout(#pipeline_layout) ordinal(2) : i32
  %p3 = hal.interface.constant.load layout(#pipeline_layout) ordinal(3) : i32
  %x0 = arith.index_castui %p0 : i32 to index
  %x1 = arith.index_castui %p1 : i32 to index
  %x2 = arith.index_castui %p2 : i32 to index
  %x3 = arith.index_castui %p3 : i32 to index
  %a = iree_tensor_ext.dispatch.workload.ordinal %x0, 0 : index
  %b = iree_tensor_ext.dispatch.workload.ordinal %x1, 1 : index
  %c = iree_tensor_ext.dispatch.workload.ordinal %x2, 2 : index
  %d = iree_tensor_ext.dispatch.workload.ordinal %x3, 3 : index
  %ab = arith.addi %a, %b : index
  %cd = arith.addi %c, %d : index
  %abcd = arith.addi %ab, %cd : index
  iree_codegen.workgroup_count_hint(%abcd, 1, 1)
  return
}

// -----

// A public function with no `translation_info` and no callees with
// translation_info is treated as a non-entry-point helper. No stub
// `dispatch_config` is created for it.

// CHECK-LABEL: func.func @helper_without_translation_info
//   CHECK-NOT: iree_codegen.dispatch_config @helper_without_translation_info
func.func @helper_without_translation_info() {
  return
}

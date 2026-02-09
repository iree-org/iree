// RUN: iree-opt --split-input-file --pass-pipeline='builtin.module(iree-rocdl-add-lds-alias-scopes)' %s | FileCheck %s

// CHECK-LABEL: llvm.func @single_lds_buffer
// CHECK: rocdl.load.to.lds
// CHECK-NOT: alias_scopes
// CHECK: rocdl.load.to.lds
// CHECK-NOT: alias_scopes
llvm.func @single_lds_buffer(
    %global : !llvm.ptr<1>,
    %desc : !llvm.struct<(ptr<3>, ptr<3>, i64, array<2 x i64>, array<2 x i64>)>) {
  %base = llvm.extractvalue %desc[1]
    : !llvm.struct<(ptr<3>, ptr<3>, i64, array<2 x i64>, array<2 x i64>)>
  %c0 = llvm.mlir.constant(0 : i64) : i64
  %c64 = llvm.mlir.constant(64 : i64) : i64
  %ptr0 = llvm.getelementptr %base[%c0] : (!llvm.ptr<3>, i64) -> !llvm.ptr<3>, f32
  %ptr1 = llvm.getelementptr %base[%c64] : (!llvm.ptr<3>, i64) -> !llvm.ptr<3>, f32
  rocdl.load.to.lds %global, %ptr0, 4, 0, 0 : <1>
  rocdl.load.to.lds %global, %ptr1, 4, 0, 0 : <1>
  llvm.return
}

// -----

// CHECK-LABEL: llvm.func @two_lds_buffers
// CHECK: rocdl.load.to.lds {{.*}} {alias_scopes = [#[[SCOPE_A:.*]]], noalias_scopes = [#[[SCOPE_B:.*]]]}
// CHECK: rocdl.load.to.lds {{.*}} {alias_scopes = [#[[SCOPE_B]]], noalias_scopes = [#[[SCOPE_A]]]}
llvm.func @two_lds_buffers(
    %global : !llvm.ptr<1>,
    %desc_A : !llvm.struct<(ptr<3>, ptr<3>, i64, array<2 x i64>, array<2 x i64>)>,
    %desc_B : !llvm.struct<(ptr<3>, ptr<3>, i64, array<2 x i64>, array<2 x i64>)>) {
  %base_A = llvm.extractvalue %desc_A[1]
    : !llvm.struct<(ptr<3>, ptr<3>, i64, array<2 x i64>, array<2 x i64>)>
  %base_B = llvm.extractvalue %desc_B[1]
    : !llvm.struct<(ptr<3>, ptr<3>, i64, array<2 x i64>, array<2 x i64>)>
  %c0 = llvm.mlir.constant(0 : i64) : i64
  %ptr_A = llvm.getelementptr %base_A[%c0] : (!llvm.ptr<3>, i64) -> !llvm.ptr<3>, f32
  %ptr_B = llvm.getelementptr %base_B[%c0] : (!llvm.ptr<3>, i64) -> !llvm.ptr<3>, f32
  rocdl.load.to.lds %global, %ptr_A, 4, 0, 0 : <1>
  rocdl.load.to.lds %global, %ptr_B, 4, 0, 0 : <1>
  llvm.return
}

// -----

// CHECK-LABEL: llvm.func @three_lds_buffers
// CHECK: rocdl.load.to.lds {{.*}} {alias_scopes = [#[[S0:.*]]], noalias_scopes = [#[[S1:.*]], #[[S2:.*]]]}
// CHECK: rocdl.load.to.lds {{.*}} {alias_scopes = [#[[S1]]], noalias_scopes = [#[[S0]], #[[S2]]]}
// CHECK: rocdl.load.to.lds {{.*}} {alias_scopes = [#[[S2]]], noalias_scopes = [#[[S0]], #[[S1]]]}
llvm.func @three_lds_buffers(
    %global : !llvm.ptr<1>,
    %desc_A : !llvm.struct<(ptr<3>, ptr<3>, i64, array<2 x i64>, array<2 x i64>)>,
    %desc_B : !llvm.struct<(ptr<3>, ptr<3>, i64, array<2 x i64>, array<2 x i64>)>,
    %desc_C : !llvm.struct<(ptr<3>, ptr<3>, i64, array<2 x i64>, array<2 x i64>)>) {
  %base_A = llvm.extractvalue %desc_A[1]
    : !llvm.struct<(ptr<3>, ptr<3>, i64, array<2 x i64>, array<2 x i64>)>
  %base_B = llvm.extractvalue %desc_B[1]
    : !llvm.struct<(ptr<3>, ptr<3>, i64, array<2 x i64>, array<2 x i64>)>
  %base_C = llvm.extractvalue %desc_C[1]
    : !llvm.struct<(ptr<3>, ptr<3>, i64, array<2 x i64>, array<2 x i64>)>
  %c0 = llvm.mlir.constant(0 : i64) : i64
  %ptr_A = llvm.getelementptr %base_A[%c0] : (!llvm.ptr<3>, i64) -> !llvm.ptr<3>, f32
  %ptr_B = llvm.getelementptr %base_B[%c0] : (!llvm.ptr<3>, i64) -> !llvm.ptr<3>, f32
  %ptr_C = llvm.getelementptr %base_C[%c0] : (!llvm.ptr<3>, i64) -> !llvm.ptr<3>, f32
  rocdl.load.to.lds %global, %ptr_A, 4, 0, 0 : <1>
  rocdl.load.to.lds %global, %ptr_B, 4, 0, 0 : <1>
  rocdl.load.to.lds %global, %ptr_C, 4, 0, 0 : <1>
  llvm.return
}

// -----

// Nested GEP chains (GEP -> GEP -> extractvalue): simulates multi-buffered
// subviews where the slot offset and element offset produce two levels of GEP.
// Two different base structs should produce two distinct scopes.
// CHECK-LABEL: llvm.func @nested_gep_two_bases
// CHECK: rocdl.load.to.lds {{.*}} {alias_scopes = [#[[N0:.*]]], noalias_scopes = [#[[N1:.*]]]}
// CHECK: rocdl.load.to.lds {{.*}} {alias_scopes = [#[[N1]]], noalias_scopes = [#[[N0]]]}
llvm.func @nested_gep_two_bases(
    %global : !llvm.ptr<1>,
    %desc_A : !llvm.struct<(ptr<3>, ptr<3>, i64, array<2 x i64>, array<2 x i64>)>,
    %desc_B : !llvm.struct<(ptr<3>, ptr<3>, i64, array<2 x i64>, array<2 x i64>)>) {
  %base_A = llvm.extractvalue %desc_A[1]
    : !llvm.struct<(ptr<3>, ptr<3>, i64, array<2 x i64>, array<2 x i64>)>
  %base_B = llvm.extractvalue %desc_B[1]
    : !llvm.struct<(ptr<3>, ptr<3>, i64, array<2 x i64>, array<2 x i64>)>
  %slot_offset = llvm.mlir.constant(4096 : i64) : i64
  %elem_offset = llvm.mlir.constant(42 : i64) : i64

  %a_slot = llvm.getelementptr %base_A[%slot_offset] : (!llvm.ptr<3>, i64) -> !llvm.ptr<3>, f32
  %a_elem = llvm.getelementptr %a_slot[%elem_offset] : (!llvm.ptr<3>, i64) -> !llvm.ptr<3>, f32

  %b_slot = llvm.getelementptr %base_B[%slot_offset] : (!llvm.ptr<3>, i64) -> !llvm.ptr<3>, f32
  %b_elem = llvm.getelementptr %b_slot[%elem_offset] : (!llvm.ptr<3>, i64) -> !llvm.ptr<3>, f32
  rocdl.load.to.lds %global, %a_elem, 4, 0, 0 : <1>
  rocdl.load.to.lds %global, %b_elem, 4, 0, 0 : <1>
  llvm.return
}

// -----

// Multi-buffered scenario: four loads targeting two slot-offset pairs from two
// different base allocations. Each slot produces a distinct descriptor after
// memref-to-LLVM lowering, so we expect 4 distinct scopes.
// CHECK-LABEL: llvm.func @multi_buffered_four_scopes
// CHECK: rocdl.load.to.lds {{.*}} {alias_scopes = [#[[MA:.*]]], noalias_scopes = [#[[MB:.*]], #[[MC:.*]], #[[MD:.*]]]}
// CHECK: rocdl.load.to.lds {{.*}} {alias_scopes = [#[[MB]]], noalias_scopes = [#[[MA]], #[[MC]], #[[MD]]]}
// CHECK: rocdl.load.to.lds {{.*}} {alias_scopes = [#[[MC]]], noalias_scopes = [#[[MA]], #[[MB]], #[[MD]]]}
// CHECK: rocdl.load.to.lds {{.*}} {alias_scopes = [#[[MD]]], noalias_scopes = [#[[MA]], #[[MB]], #[[MC]]]}
llvm.func @multi_buffered_four_scopes(
    %global : !llvm.ptr<1>,
    %desc_A0 : !llvm.struct<(ptr<3>, ptr<3>, i64, array<2 x i64>, array<2 x i64>)>,
    %desc_B0 : !llvm.struct<(ptr<3>, ptr<3>, i64, array<2 x i64>, array<2 x i64>)>,
    %desc_A1 : !llvm.struct<(ptr<3>, ptr<3>, i64, array<2 x i64>, array<2 x i64>)>,
    %desc_B1 : !llvm.struct<(ptr<3>, ptr<3>, i64, array<2 x i64>, array<2 x i64>)>) {
  %c0 = llvm.mlir.constant(0 : i64) : i64

  %base_A0 = llvm.extractvalue %desc_A0[1]
    : !llvm.struct<(ptr<3>, ptr<3>, i64, array<2 x i64>, array<2 x i64>)>
  %ptr_A0 = llvm.getelementptr %base_A0[%c0] : (!llvm.ptr<3>, i64) -> !llvm.ptr<3>, f32
  %base_B0 = llvm.extractvalue %desc_B0[1]
    : !llvm.struct<(ptr<3>, ptr<3>, i64, array<2 x i64>, array<2 x i64>)>
  %ptr_B0 = llvm.getelementptr %base_B0[%c0] : (!llvm.ptr<3>, i64) -> !llvm.ptr<3>, f32

  %base_A1 = llvm.extractvalue %desc_A1[1]
    : !llvm.struct<(ptr<3>, ptr<3>, i64, array<2 x i64>, array<2 x i64>)>
  %ptr_A1 = llvm.getelementptr %base_A1[%c0] : (!llvm.ptr<3>, i64) -> !llvm.ptr<3>, f32
  %base_B1 = llvm.extractvalue %desc_B1[1]
    : !llvm.struct<(ptr<3>, ptr<3>, i64, array<2 x i64>, array<2 x i64>)>
  %ptr_B1 = llvm.getelementptr %base_B1[%c0] : (!llvm.ptr<3>, i64) -> !llvm.ptr<3>, f32
  rocdl.load.to.lds %global, %ptr_A0, 4, 0, 0 : <1>
  rocdl.load.to.lds %global, %ptr_B0, 4, 0, 0 : <1>
  rocdl.load.to.lds %global, %ptr_A1, 4, 0, 0 : <1>
  rocdl.load.to.lds %global, %ptr_B1, 4, 0, 0 : <1>
  llvm.return
}

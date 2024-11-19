// RUN: iree-opt %s --split-input-file --verify-diagnostics \
// RUN:   --pass-pipeline="builtin.module(func.func(iree-codegen-verify-workgroup-distribution))" \
// RUN:   | FileCheck %s

// expected-error@+1 {{op failed on workgroup distribution verification}}
func.func @write_outside_workgroup_forall(%i: i32, %out: memref<32xi32, #hal.descriptor_type<storage_buffer>>) {
  scf.forall (%arg0) in (32) {
  } {mapping = [#iree_codegen.workgroup_mapping<x>]}
  %c0 = arith.constant 0 : index
  // expected-error@+1 {{write affecting operations on global resources are restricted to workgroup distributed contexts.}}
  memref.store %i, %out[%c0] : memref<32xi32, #hal.descriptor_type<storage_buffer>>
  return
}

// -----

// CHECK: func @non_workgroup_write_outside_workgroup_forall
func.func @non_workgroup_write_outside_workgroup_forall(
  %i: i32, %out: memref<32xi32, #hal.descriptor_type<storage_buffer>>, %out2: memref<32xi32>) {
  scf.forall (%arg0) in (32) {
    memref.store %i, %out[%arg0] : memref<32xi32, #hal.descriptor_type<storage_buffer>>
  } {mapping = [#iree_codegen.workgroup_mapping<x>]}
  %c0 = arith.constant 0 : index
  memref.store %i, %out2[%c0] : memref<32xi32>
  return
}

// -----

// expected-error@+1 {{op failed on workgroup distribution verification}}
func.func @write_nested_in_other_forall(%i: i32, %out: memref<32xi32, #hal.descriptor_type<storage_buffer>>) {
  scf.forall (%arg0) in (32) {
  } {mapping = [#iree_codegen.workgroup_mapping<x>]}
  %c0 = arith.constant 0 : index
  scf.forall (%arg1) in (32) {
    // expected-error@+1 {{write affecting operations on global resources are restricted to workgroup distributed contexts.}}
    memref.store %i, %out[%arg1] : memref<32xi32, #hal.descriptor_type<storage_buffer>>
  }
  return
}

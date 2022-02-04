// RUN: iree-opt -allow-unregistered-dialect -iree-convert-to-llvm -split-input-file %s | FileCheck %s

llvm.func @sink(i64)

// CHECK-LABEL: llvm.func internal @workgroup_id
func @workgroup_id() {
  // CHECK: %[[PTR:.+]] = llvm.load %arg1 : !llvm.ptr<array<3 x i32>>
  // CHECK: %[[Z32:.+]] = llvm.extractvalue %[[PTR]][2] : !llvm.array<3 x i32>
  // CHECK: %[[Z64:.+]] = llvm.zext %[[Z32]] : i32 to i64
  %workgroup_id_z = hal.interface.workgroup.id[2] : index
  // CHECK-NEXT: llvm.call @sink(%[[Z64]])
  %val = arith.index_cast %workgroup_id_z : index to i64
  llvm.call @sink(%val) : (i64) -> ()
  return
}

// -----

llvm.func @sink(i64)

// CHECK-LABEL: llvm.func internal @workgroup_size
func @workgroup_size() {
  // CHECK: %[[STATE:.+]] = llvm.load %arg0 : !llvm.ptr<struct<"iree_hal_executable_dispatch_state_v0_t"
  // CHECK: %[[SIZE_PTR:.+]] = llvm.extractvalue %[[STATE]][1] : !llvm.struct<"iree_hal_executable_dispatch_state_v0_t"
  // CHECK: %[[Z32:.+]] = llvm.extractvalue %[[SIZE_PTR]][2] : !llvm.array<3 x i32>
  // CHECK: %[[Z64:.+]] = llvm.zext %[[Z32]] : i32 to i64
  %workgroup_size_z = hal.interface.workgroup.size[2] : index
  // CHECK-NEXT: llvm.call @sink(%[[Z64]])
  %val = arith.index_cast %workgroup_size_z : index to i64
  llvm.call @sink(%val) : (i64) -> ()
  return
}

// -----

llvm.func @sink(i64)

// CHECK-LABEL: llvm.func internal @workgroup_count
func @workgroup_count() {
  // CHECK: %[[STATE:.+]] = llvm.load %arg0 : !llvm.ptr<struct<"iree_hal_executable_dispatch_state_v0_t"
  // CHECK: %[[COUNT_PTR:.+]] = llvm.extractvalue %[[STATE]][0] : !llvm.struct<"iree_hal_executable_dispatch_state_v0_t"
  // CHECK: %[[Z32:.+]] = llvm.extractvalue %[[COUNT_PTR]][2] : !llvm.array<3 x i32>
  // CHECK: %[[Z64:.+]] = llvm.zext %[[Z32]] : i32 to i64
  %workgroup_count_z = hal.interface.workgroup.count[2] : index
  // CHECK-NEXT: llvm.call @sink(%[[Z64]])
  %val = arith.index_cast %workgroup_count_z : index to i64
  llvm.call @sink(%val) : (i64) -> ()
  return
}

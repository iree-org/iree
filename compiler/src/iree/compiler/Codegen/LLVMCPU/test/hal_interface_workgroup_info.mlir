// RUN: iree-opt --iree-convert-to-llvm --split-input-file %s | FileCheck %s

// CHECK-LABEL: llvm.func @workgroup_id
func.func @workgroup_id() {
  // CHECK: %[[STATE:.+]] = llvm.load %arg2 : !llvm.ptr -> !llvm.struct<"iree_hal_executable_workgroup_state_v0_t"
  // CHECK: %[[Z16:.+]] = llvm.extractvalue %[[STATE]][2]
  // CHECK: %[[Z64:.+]] = llvm.zext %[[Z16]] : i16 to i64
  %workgroup_id_z = hal.interface.workgroup.id[2] : index
  // CHECK: llvm.call @sink(%[[Z64]])
  %val = arith.index_cast %workgroup_id_z : index to i64
  llvm.call @sink(%val) : (i64) -> ()
  return
}
llvm.func @sink(%arg0: i64) {
  llvm.return
}

// -----

// CHECK-LABEL: llvm.func @workgroup_size
func.func @workgroup_size() {
  // CHECK: %[[STATE:.+]] = llvm.load %arg1 : !llvm.ptr -> !llvm.struct<"iree_hal_executable_dispatch_state_v0_t"
  // CHECK: %[[Z16:.+]] = llvm.extractvalue %[[STATE]][2]
  // CHECK: %[[Z64:.+]] = llvm.zext %[[Z16]] : i16 to i64
  %workgroup_size_z = hal.interface.workgroup.size[2] : index
  // CHECK: llvm.call @sink(%[[Z64]])
  %val = arith.index_cast %workgroup_size_z : index to i64
  llvm.call @sink(%val) : (i64) -> ()
  return
}
llvm.func @sink(%arg0: i64) {
  llvm.return
}

// -----

// CHECK-LABEL: llvm.func @workgroup_count
func.func @workgroup_count() {
  // CHECK: %[[STATE:.+]] = llvm.load %arg1 : !llvm.ptr -> !llvm.struct<"iree_hal_executable_dispatch_state_v0_t"
  // CHECK: %[[Z16:.+]] = llvm.extractvalue %[[STATE]][6]
  // CHECK: %[[Z64:.+]] = llvm.zext %[[Z16]] : i16 to i64
  %workgroup_count_z = hal.interface.workgroup.count[2] : index
  // CHECK: llvm.call @sink(%[[Z64]])
  %val = arith.index_cast %workgroup_count_z : index to i64
  llvm.call @sink(%val) : (i64) -> ()
  return
}
llvm.func @sink(%arg0: i64) {
  llvm.return
}

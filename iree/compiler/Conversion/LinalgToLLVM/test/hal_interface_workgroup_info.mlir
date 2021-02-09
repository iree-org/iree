// RUN: iree-opt -allow-unregistered-dialect -iree-codegen-convert-to-llvm -split-input-file %s | IreeFileCheck %s

// CHECK-LABEL: llvm.func @workgroup_id
func @workgroup_id() {
  // CHECK: %[[PTR:.+]] = llvm.load %arg2 : !llvm.ptr<array<3 x i32>>
  // CHECK: %[[Z32:.+]] = llvm.extractvalue %[[PTR]][2 : i32] : !llvm.array<3 x i32>
  // CHECK: %[[Z64:.+]] = llvm.zext %[[Z32]] : i32 to i64
  %workgroup_id_z = hal.interface.workgroup.id[2] : index
  // CHECK-NEXT: "test.sink"(%[[Z64]])
  "test.sink"(%workgroup_id_z) : (index) -> ()
  return
}

// -----

// CHECK-LABEL: llvm.func @workgroup_size
func @workgroup_size() {
  // CHECK: %[[PTR:.+]] = llvm.load %arg4 : !llvm.ptr<array<3 x i32>>
  // CHECK: %[[Z32:.+]] = llvm.extractvalue %[[PTR]][2 : i32] : !llvm.array<3 x i32>
  // CHECK: %[[Z64:.+]] = llvm.zext %[[Z32]] : i32 to i64
  %workgroup_size_z = hal.interface.workgroup.size[2] : index
  // CHECK-NEXT: "test.sink"(%[[Z64]])
  "test.sink"(%workgroup_size_z) : (index) -> ()
  return
}

// -----

// CHECK-LABEL: llvm.func @workgroup_count
func @workgroup_count() {
  // CHECK: %[[PTR:.+]] = llvm.load %arg3 : !llvm.ptr<array<3 x i32>>
  // CHECK: %[[Z32:.+]] = llvm.extractvalue %[[PTR]][2 : i32] : !llvm.array<3 x i32>
  // CHECK: %[[Z64:.+]] = llvm.zext %[[Z32]] : i32 to i64
  %workgroup_count_z = hal.interface.workgroup.count[2] : index
  // CHECK-NEXT: "test.sink"(%[[Z64]])
  "test.sink"(%workgroup_count_z) : (index) -> ()
  return
}

// RUN: iree-opt --split-input-file --iree-vm-reify-rodata-tables %s | FileCheck %s

// CHECK: #[[$DATA:.+]] = #util.composite<4xi8, [
#table_data = #util.composite<4xi8, [
// CHECK-NEXT: dense<0> : vector<1xi8>,
  dense<0> : vector<1xi8>,
// CHECK-NEXT: dense<1> : vector<1xi8>,
  dense<1> : vector<1xi8>,
// CHECK-NEXT: dense<[2, 3]> : vector<2xi8>,
  dense<[2, 3]> : vector<2xi8>,
]>

vm.module @module {
  // CHECK-LABEL: vm.func @fn
  vm.func @fn() {
    // CHECK-DAG: = vm.rodata.inline : !vm.buffer = dense<[0, 1, 1, 1, 2, 2]> : vector<6xi32>
    // CHECK-DAG: = vm.rodata.inline : !vm.buffer = #[[$DATA]]
    %0:2 = vm.rodata.table : !vm.buffer, !vm.buffer = #table_data
    vm.return
  }
}

// -----

// CHECK: #[[$DATA:.+]] = #util.composite<2xi8, [
#table_data = #util.composite<2xi8, [
// CHECK-NEXT: dense<[2, 3]> : vector<2xi8>,
  dense<[2, 3]> : vector<2xi8>,
]>

vm.module @module {
  // CHECK-LABEL: vm.func @fn
  vm.func @fn() {
    // CHECK-DAG: = vm.rodata.inline "table" {alignment = 64 : i64} : !vm.buffer = dense<[0, 2]>
    // CHECK-DAG: = vm.rodata.inline "data" {alignment = 64 : i64} : !vm.buffer = #[[$DATA]]
    %0:2 = vm.rodata.table {table_name = "table", data_name = "data", alignment = 64 : i64} : !vm.buffer, !vm.buffer = #table_data
    vm.return
  }
}

// -----

// CHECK: #[[$DATA:.+]] = #util.composite<15xi8, [
#table_data = #util.composite<12xi8, [
// CHECK-NEXT: dense<[2, 3]> : vector<2xi8>,
// CHECK-NEXT: dense<0> : vector<1xi8>,
  dense<[2, 3]> : vector<2xi8>,
// CHECK-NEXT: "hello",
// CHECK-NEXT: dense<0> : vector<1xi8>,
  "hello",
// CHECK-NEXT: "world",
// CHECK-NEXT: dense<0> : vector<1xi8>,
  "world",
]>

vm.module @module {
  // CHECK-LABEL: vm.func @fn
  vm.func @fn() {
    // CHECK-DAG: = vm.rodata.inline : !vm.buffer = dense<[0, 2, 3, 5, 9, 5]>
    // CHECK-DAG: = vm.rodata.inline : !vm.buffer = #[[$DATA]]
    %0:2 = vm.rodata.table {data_alignment = 3 : i64} : !vm.buffer, !vm.buffer = #table_data
    vm.return
  }
}

// Tests printing and parsing of executable/structural ops.

// RUN: iree-opt -split-input-file %s | iree-opt -split-input-file | IreeFileCheck %s

// CHECK-LABEL: @ex
hal.executable @ex {
  // CHECK-NEXT: hal.executable.entry_point @entry0 attributes {
  // CHECK-SAME:     ordinal = 0 : i32
  // CHECK-SAME:     workgroup_size = dense<[4, 1, 1]> : vector<3xi32>
  hal.executable.entry_point @entry0 attributes {
    ordinal = 0 : i32,
    workgroup_size = dense<[4, 1, 1]> : vector<3xi32>
  }
  // CHECK-NEXT: hal.executable.binary
  hal.executable.binary attributes {
    // CHECK-SAME: data = dense<1> : vector<128xi8>,
    data = dense<1> : vector<128xi8>,
    // CHECK-SAME: format = 1230128453 : i32
    format = 1230128453 : i32
  }
}

// -----

// CHECK-LABEL: @ex_with_source
hal.executable @ex_with_source {
  // CHECK-NEXT: hal.executable.binary
  hal.executable.binary attributes {
    // CHECK-SAME: data = dense<1> : vector<128xi8>,
    data = dense<1> : vector<128xi8>,
    // CHECK-SAME: format = 1230128453 : i32
    format = 1230128453 : i32
  } {
    // CHECK-NEXT: module {
    module {
      // CHECK-NEXT: func @dispatch0
      func @dispatch0(%arg0: memref<4xf32>, %arg1: memref<4xf32>) attributes {
          iree.executable.export,
          iree.ordinal = 0 : i32} {
        %0 = "iree_ll_interp.alloc_heap"() : () -> memref<4xf32>
        "iree_ll_interp.add_f"(%arg0, %arg0, %0) : (memref<4xf32>, memref<4xf32>, memref<4xf32>) -> ()
        %1 = "iree_ll_interp.constant"() {value = dense<0> : tensor<1xi64>} : () -> memref<1xi64>
        %2 = "iree_ll_interp.constant"() {value = dense<4> : tensor<1xi64>} : () -> memref<1xi64>
        "iree_ll_interp.dynamic_copy"(%0, %1, %arg1, %1, %2) : (memref<4xf32>, memref<1xi64>, memref<4xf32>, memref<1xi64>, memref<1xi64>) -> ()
        return
      }
    }
  }
}

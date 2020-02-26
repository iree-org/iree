// Tests printing and parsing of executable/structural ops.

// RUN: iree-opt -split-input-file %s | iree-opt -split-input-file | IreeFileCheck %s

// CHECK-LABEL: @interface_io
func @interface_io() {
  %c16_i32 = constant 16 : i32
  // CHECK: [[ARG0:%.+]] = hal.interface.load.tensor @interface::@s0b0, offset = %c16_i32 : tensor<4xf32>
  %arg0 = hal.interface.load.tensor @interface::@s0b0, offset=%c16_i32 : tensor<4xf32>
  // CHECK-NEXT: [[TEMP:%.+]] = xla_hlo.add [[ARG0]], [[ARG0]]
  %0 = xla_hlo.add %arg0, %arg0 : tensor<4xf32>
  %c32_i32 = constant 32 : i32
  // CHECK: hal.interface.store.tensor [[TEMP]], @interface::@s0b1, offset = %c32_i32 : tensor<4xf32>
  hal.interface.store.tensor %0, @interface::@s0b1, offset=%c32_i32 : tensor<4xf32>
  return
}

// -----

// CHECK-LABEL: @ex
hal.executable @ex {
  // CHECK-DAG: hal.executable.entry_point @entry0 attributes {
  // CHECK-SAME:     interface = @interface
  // CHECK-SAME:     ordinal = 0 : i32
  // CHECK-SAME:     signature = (tensor<4xf32>) -> tensor<4xf32>
  // CHECK-SAME:     workgroup_size = dense<[4, 1, 1]> : vector<3xi32>
  hal.executable.entry_point @entry0 attributes {
    interface = @interface,
    ordinal = 0 : i32,
    signature = (tensor<4xf32>) -> tensor<4xf32>,
    workgroup_size = dense<[4, 1, 1]> : vector<3xi32>
  }
  // CHECK-DAG: hal.interface @interface
  hal.interface @interface {
    // CHECK-NEXT: hal.interface.binding @s0b0, set=0, binding=0, type="StorageBuffer", access="Read"
    hal.interface.binding @s0b0, set=0, binding=0, type="StorageBuffer", access="Read"
    // CHECK-NEXT: hal.interface.binding @s0b1, set=0, binding=1, type="StorageBuffer", access="Read|Write"
    hal.interface.binding @s0b1, set=0, binding=1, type="StorageBuffer", access="Read|Write"
  }
  // CHECK: hal.executable.binary
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

// Tests printing and parsing of executable/structural ops.

// RUN: iree-opt -allow-unregistered-dialect -split-input-file %s | iree-opt -allow-unregistered-dialect -split-input-file | IreeFileCheck %s

// CHECK-LABEL: @ex
hal.executable @ex {
  // CHECK: hal.executable.target @backend, filter="backend"
  hal.executable.target @backend, filter="backend" {
    // CHECK-DAG: hal.executable.entry_point @entry0 attributes {
    // CHECK-SAME:     interface = @interface
    // CHECK-SAME:     ordinal = 0 : i32
    // CHECK-SAME:     signature = (tensor<4xf32>) -> tensor<4xf32>
    // CHECK-SAME:     workgroup_size = [4 : index, 1 : index, 1 : index]
    hal.executable.entry_point @entry0 attributes {
      interface = @interface,
      ordinal = 0 : i32,
      signature = (tensor<4xf32>) -> tensor<4xf32>,
      workgroup_size = [4 : index, 1 : index, 1 : index]
    }
  }
  // CHECK-DAG: hal.interface @interface
  hal.interface @interface {
    // CHECK-NEXT: hal.interface.binding @s0b0, set=0, binding=0, type="StorageBuffer", access="Read"
    hal.interface.binding @s0b0, set=0, binding=0, type="StorageBuffer", access="Read"
    // CHECK-NEXT: hal.interface.binding @s0b1, set=0, binding=1, type="StorageBuffer", access="Read|Write"
    hal.interface.binding @s0b1, set=0, binding=1, type="StorageBuffer", access="Read|Write"
  }
  // CHECK: hal.executable.binary
  hal.executable.binary @backend_binary attributes {
    // CHECK-SAME: data = dense<1> : vector<128xi8>,
    data = dense<1> : vector<128xi8>,
    // CHECK-SAME: format = 1230128453 : i32
    format = 1230128453 : i32
  }
}

// -----

// CHECK-LABEL: @ex_with_workgroup_count_region
hal.executable @ex_with_workgroup_count_region {
  // CHECK: hal.executable.target @backend, filter="backend"
  hal.executable.target @backend, filter="backend" {
    // CHECK-DAG: hal.executable.entry_point @entry0 attributes {
    // CHECK-SAME:     interface = @interface
    // CHECK-SAME:     ordinal = 0 : i32
    // CHECK-SAME:     signature = (tensor<4xf32>) -> tensor<4xf32>
    // CHECK-SAME:     workgroup_size = [4 : index, 1 : index, 1 : index]
    hal.executable.entry_point @entry0 attributes {
      interface = @interface,
      ordinal = 0 : i32,
      signature = (tensor<4xf32>) -> tensor<4xf32>,
      workgroup_size = [4 : index, 1 : index, 1 : index]
    } {
    ^bb0(%arg0: index, %arg1: index, %arg2: index):
      hal.return %arg0, %arg1, %arg2 : index, index, index
    }
  }
  // CHECK-DAG: hal.interface @interface
  hal.interface @interface {
    // CHECK-NEXT: hal.interface.binding @s0b0, set=0, binding=0, type="StorageBuffer", access="Read"
    hal.interface.binding @s0b0, set=0, binding=0, type="StorageBuffer", access="Read"
    // CHECK-NEXT: hal.interface.binding @s0b1, set=0, binding=1, type="StorageBuffer", access="Read|Write"
    hal.interface.binding @s0b1, set=0, binding=1, type="StorageBuffer", access="Read|Write"
  }
  // CHECK: hal.executable.binary
  hal.executable.binary @backend_binary attributes {
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
  hal.executable.binary @backend_binary attributes {
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

// -----

// CHECK-LABEL: @executable_create
// CHECK-SAME: %[[DEVICE:.+]]: !hal.device,
// CHECK-SAME: %[[LAYOUT0:.+]]: !hal.executable_layout,
// CHECK-SAME: %[[LAYOUT1:.+]]: !hal.executable_layout
func @executable_create(%device : !hal.device, %layout0 : !hal.executable_layout, %layout1 : !hal.executable_layout) {
  // CHECK: = hal.executable.create %[[DEVICE]], @exe::@binary1, layouts = [%[[LAYOUT0]], %[[LAYOUT1]]] : !hal.executable
  %0 = hal.executable.create %device, @exe::@binary1, layouts = [%layout0, %layout1] : !hal.executable
  return
}

// -----

// CHECK-LABEL: @executable_layout_create
func @executable_layout_create(%arg0 : !hal.device, %arg1 : !hal.descriptor_set_layout) {
  // CHECK: hal.executable_layout.create %arg0, push_constants = 1, set_layouts = [%arg1] : !hal.executable_layout
  %executable_layout = hal.executable_layout.create %arg0, push_constants = 1, set_layouts = [%arg1] : !hal.executable_layout
  return
}

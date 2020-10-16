// RUN: iree-opt -split-input-file -iree-hal-canonicalize-interfaces %s | IreeFileCheck %s

// CHECK-LABEL: @executable_0
hal.executable @executable_0 {
  // CHECK-NEXT: hal.interface @_canonical_interface attributes
  // CHECK-SAME:     push_constants = 4 : i32
  // CHECK-NEXT:   hal.interface.binding @s0b0, set=0, binding=0, type="StorageBuffer", access="Read"
  // CHECK-NEXT:   hal.interface.binding @s0b1, set=0, binding=1, type="StorageBuffer", access="Read"
  // CHECK-NEXT:   hal.interface.binding @s0b2, set=0, binding=2, type="StorageBuffer", access="Read"
  // CHECK-NEXT:   hal.interface.binding @s0b3, set=0, binding=3, type="StorageBuffer", access="Write"
  // CHECK-NEXT:   hal.interface.binding @s0b4, set=0, binding=4, type="StorageBuffer", access="Write|Discard"
  hal.interface @interface_0 attributes {push_constants = 1 : i32} {
    hal.interface.binding @arg0, set=0, binding=0, type="StorageBuffer", access="Read"
    hal.interface.binding @ret0, set=0, binding=1, type="StorageBuffer", access="Write|Discard"
  }
  hal.interface @interface_1 {
    hal.interface.binding @arg0, set=0, binding=0, type="StorageBuffer", access="Read"
    hal.interface.binding @arg1, set=0, binding=1, type="StorageBuffer", access="Read"
    hal.interface.binding @arg2, set=0, binding=2, type="StorageBuffer", access="Read"
    hal.interface.binding @ret0, set=0, binding=3, type="StorageBuffer", access="Write|Discard"
  }
  // CHECK: hal.executable.target @backend_0
  hal.executable.target @backend_0, filter="vmla" {
    // CHECK: hal.executable.entry_point @entry_fn
    // CHECK-SAME: interface = @_canonical_interface
    hal.executable.entry_point @entry_fn attributes {interface = @interface_0, ordinal = 0 : i32, signature = (tensor<384x512xf32>) -> tensor<384x512xf32>}
    module {
      func @entry_fn() {
        %c0 = constant 0 : index
        // CHECK: hal.interface.load.tensor @_canonical_interface::@s0b0, offset = %c0 : tensor<384x512xf32>
        %0 = hal.interface.load.tensor @interface_0::@arg0, offset = %c0 : tensor<384x512xf32>
        // CHECK: hal.interface.store.tensor %0, @_canonical_interface::@s0b4, offset = %c0 : tensor<384x512xf32>
        hal.interface.store.tensor %0, @interface_0::@ret0, offset = %c0 : tensor<384x512xf32>
        return
      }
    }
  }
  // CHECK: hal.executable.target @backend_1
  hal.executable.target @backend_1, filter="vulkan" {
    // CHECK: hal.executable.entry_point @entry_fn
    // CHECK-SAME: interface = @_canonical_interface
    hal.executable.entry_point @entry_fn attributes {interface = @interface_1, ordinal = 1 : i32, signature = (tensor<384x512xf32>, tensor<384x512xf32>, tensor<384x512xf32>) -> tensor<384x512xf32>}
    module {
      func @entry_fn() {
        %c0 = constant 0 : index
        // CHECK: hal.interface.load.tensor @_canonical_interface::@s0b0, offset = %c0 : tensor<384x512xf32>
        %0 = hal.interface.load.tensor @interface_1::@arg0, offset = %c0 : tensor<384x512xf32>
        // CHECK: hal.interface.load.tensor @_canonical_interface::@s0b1, offset = %c0 : tensor<384x512xf32>
        %1 = hal.interface.load.tensor @interface_1::@arg1, offset = %c0 : tensor<384x512xf32>
        // CHECK: hal.interface.load.tensor @_canonical_interface::@s0b2, offset = %c0 : tensor<384x512xf32>
        %2 = hal.interface.load.tensor @interface_1::@arg2, offset = %c0 : tensor<384x512xf32>
        // CHECK: hal.interface.store.tensor %2, @_canonical_interface::@s0b4, offset = %c0 : tensor<384x512xf32>
        hal.interface.store.tensor %2, @interface_1::@ret0, offset = %c0 : tensor<384x512xf32>
        return
      }
    }
  }
}
// CHECK: @executable_1
hal.executable @executable_1 {
  // CHECK-NEXT: hal.interface @_canonical_interface
  hal.interface @interface_2 attributes {push_constants = 4 : i32} {
    hal.interface.binding @arg0, set=0, binding=0, type="StorageBuffer", access="Read"
    hal.interface.binding @arg1, set=0, binding=1, type="StorageBuffer", access="Read"
    hal.interface.binding @ret0, set=0, binding=2, type="StorageBuffer", access="Write|Discard"
  }
  hal.interface @interface_3 {
    hal.interface.binding @arg0, set=0, binding=0, type="StorageBuffer", access="Read"
    hal.interface.binding @arg1, set=0, binding=1, type="StorageBuffer", access="Read"
    hal.interface.binding @ret0, set=0, binding=2, type="StorageBuffer", access="Write"
    hal.interface.binding @ret1, set=0, binding=3, type="StorageBuffer", access="Write|Discard"
  }
  // CHECK: hal.executable.target @backend_0
  hal.executable.target @backend_0, filter="vmla" {
    // CHECK: hal.executable.entry_point @entry_fn
    // CHECK-SAME: interface = @_canonical_interface
    hal.executable.entry_point @entry_fn attributes {interface = @interface_2, ordinal = 0 : i32, signature = (tensor<384x512xf32>, tensor<384x512xf32>) -> tensor<384x512xf32>}
    module {
      func @entry_fn() {
        %c0 = constant 0 : index
        // CHECK: hal.interface.load.tensor @_canonical_interface::@s0b0, offset = %c0 : tensor<384x512xf32>
        %0 = hal.interface.load.tensor @interface_2::@arg0, offset = %c0 : tensor<384x512xf32>
        // CHECK: hal.interface.load.tensor @_canonical_interface::@s0b1, offset = %c0 : tensor<384x512xf32>
        %1 = hal.interface.load.tensor @interface_2::@arg1, offset = %c0 : tensor<384x512xf32>
        // CHECK: hal.interface.store.tensor %0, @_canonical_interface::@s0b4, offset = %c0 : tensor<384x512xf32>
        hal.interface.store.tensor %0, @interface_2::@ret0, offset = %c0 : tensor<384x512xf32>
        return
      }
    }
  }
  // CHECK: hal.executable.target @backend_1
  hal.executable.target @backend_1, filter="vulkan" {
    // CHECK: hal.executable.entry_point @entry_fn
    // CHECK-SAME: interface = @_canonical_interface
    hal.executable.entry_point @entry_fn attributes {interface = @interface_3, ordinal = 1 : i32, signature = (tensor<384x512xf32>, tensor<384x512xf32>) -> (tensor<384x512xf32>, tensor<384x512xf32>)}
    module {
      func @entry_fn() {
        %c0 = constant 0 : index
        // CHECK: hal.interface.load.tensor @_canonical_interface::@s0b0, offset = %c0 : tensor<384x512xf32>
        %0 = hal.interface.load.tensor @interface_3::@arg0, offset = %c0 : tensor<384x512xf32>
        // CHECK: hal.interface.load.tensor @_canonical_interface::@s0b1, offset = %c0 : tensor<384x512xf32>
        %1 = hal.interface.load.tensor @interface_3::@arg1, offset = %c0 : tensor<384x512xf32>
        // CHECK: hal.interface.store.tensor %0, @_canonical_interface::@s0b3, offset = %c0 : tensor<384x512xf32>
        hal.interface.store.tensor %0, @interface_3::@ret0, offset = %c0 : tensor<384x512xf32>
        // CHECK: hal.interface.store.tensor %1, @_canonical_interface::@s0b4, offset = %c0 : tensor<384x512xf32>
        hal.interface.store.tensor %1, @interface_3::@ret1, offset = %c0 : tensor<384x512xf32>
        return
      }
    }
  }
}

// RUN: iree-opt -split-input-file -iree-convert-to-hal -canonicalize %s | IreeFileCheck %s

hal.executable @ex0 {
  hal.interface @interface {
    hal.interface.binding @s0b0, set=0, binding=0, type="StorageBuffer", access="Read"
    hal.interface.binding @s0b1, set=0, binding=1, type="StorageBuffer", access="Read|Write"
  }
  hal.executable.target @vmla, filter="vmla" {
    hal.executable.entry_point @entry0 attributes {
      interface = @interface,
      ordinal = 0 : i32,
      signature = (tensor<128xf32>) -> tensor<128xf32>
    }
    module {}
  }
}

// CHECK-LABEL: func @multipleDispatches
func @multipleDispatches(%arg0: tensor<128xf32>) -> tensor<128xf32> {
  // CHECK-DAG: %[[C0:.+]] = constant 0
  // CHECK-DAG: %[[C128:.+]] = constant 128
  %cst = constant 128 : index
  // CHECK: %[[RET_BUF:.+]] = hal.allocator.allocate {{.+}}, "HostVisible|DeviceVisible|DeviceLocal", "Constant|Transfer|Mapping|Dispatch"
  // CHECK: %[[TMP_BUF:.+]] = hal.allocator.allocate {{.+}}, "DeviceVisible|DeviceLocal", "Transfer|Dispatch"
  // CHECK: %[[CMD:.+]] = hal.command_buffer.create {{.+}}, OneShot, "Transfer|Dispatch"
  // CHECK-NEXT: hal.command_buffer.begin %[[CMD]]
  %0 = flow.ex.stream.fragment(%arg1 = %cst : index, %arg2 = %arg0 : tensor<128xf32>) -> tensor<128xf32> {
    //  CHECK-DAG: %[[EXE_LAYOUT:.+]] = hal.executable_layout.lookup
    //      CHECK: hal.command_buffer.push_descriptor_set %[[CMD]], %[[EXE_LAYOUT]], set = %c0, bindings = [%c0 = (%arg0, %c0, %c512), %c1 = (%[[TMP_BUF]], %c0, %c512)]
    //      CHECK: hal.command_buffer.dispatch.symbol {{.+}}, @ex0::@vmla::@entry0, workgroup_xyz
    //      CHECK: hal.command_buffer.execution_barrier
    %1 = flow.dispatch @ex0::@entry0[%arg1] (%arg2) : (tensor<128xf32>) -> tensor<128xf32>
    //      CHECK: hal.command_buffer.push_descriptor_set
    //      CHECK: hal.command_buffer.dispatch.symbol {{.+}}, @ex0::@vmla::@entry0, workgroup_xyz
    //      CHECK: hal.command_buffer.execution_barrier
    %2 = flow.dispatch @ex0::@entry0[%arg1] (%1) : (tensor<128xf32>) -> tensor<128xf32>
    flow.return %2 : tensor<128xf32>
  }
  // CHECK: hal.command_buffer.end %[[CMD]]
  // CHECK-NEXT: hal.ex.submit_and_wait {{.+}}, %[[CMD]]
  // CHECK-NEXT: return %[[RET_BUF]]
  return %0 : tensor<128xf32>
}

// -----

// CHECK-LABEL: @tensorUpdate
// CHECK-SAME: (%[[UBUF:.+]]:{{.+}}, %[[TBUF:.+]]:{{.+}})
func @tensorUpdate(%arg0 : tensor<1x1x10xf32>, %arg1 : tensor<5x1x10xf32>) -> tensor<5x1x10xf32> {
  %c4 = constant 4 : index
  %c1 = constant 1 : index
  // CHECK: %[[RET_BUF:.+]] = hal.allocator.allocate
  // CHECK: %[[CMD:.+]] = hal.command_buffer.create
  // CHECK-NEXT: hal.command_buffer.begin %[[CMD]]
  %0 = flow.ex.stream.fragment(%arg2 = %arg0 : tensor<1x1x10xf32>, %arg3 = %arg1 : tensor<5x1x10xf32>, %arg4 = %c4 : index, %arg5 = %c1 : index) -> tensor<5x1x10xf32> {
    // TODO(laurenzo): Update these checks to be more precise. The regexes can
    // match too much, masking issues.
    // CHECK-NEXT: hal.command_buffer.copy_buffer %[[CMD]], %[[TBUF]], %c0, %[[RET_BUF]], %c0, %c200
    // CHECK: hal.command_buffer.execution_barrier
    // CHECK-NEXT: hal.command_buffer.copy_buffer %[[CMD]], %[[UBUF]], %c0, %[[RET_BUF]], %c204, %c40
    %1 = flow.tensor.update %arg2, %arg3[%arg4, %arg5, %arg5] : tensor<1x1x10xf32> -> tensor<5x1x10xf32>
    flow.return %1 : tensor<5x1x10xf32>
  }
  // CHECK: hal.command_buffer.end %[[CMD]]
  // CHECK: return %[[RET_BUF]]
  return %0 : tensor<5x1x10xf32>
}

// -----

hal.executable @ex0 {
  hal.interface @interface attributes {push_constants = 2 : i32} {
    hal.interface.binding @s0b0, set=0, binding=0, type="StorageBuffer", access="Read"
    hal.interface.binding @s0b1, set=0, binding=1, type="StorageBuffer", access="Read|Write"
  }
  hal.executable.target @vmla, filter="vmla" {
    hal.executable.entry_point @entry0 attributes {
      interface = @interface,
      ordinal = 0 : i32,
      signature = (tensor<?x128xf32>, index) -> tensor<?x128xf32>
    }
    module {}
  }
}

// CHECK-LABEL: func @dispatchWithShapeTies
// CHECK-SAME: (%[[T:.+]]:{{.+}}, %[[BS:.+]]:{{.+}})
func @dispatchWithShapeTies(%arg0: tensor<?x128xf32>, %bs : index) -> tensor<?x128xf32> {
  %cst = constant 128 : index
  // Verify that size computation derives from the passed dynamic index.
  // CHECK-DAG: %[[BS4:.+]] = muli %[[BS]], %c4 : index
  // CHECK-DAG: = muli %[[BS4]], %c128 : index
  // Verify that an i32 is pushed.
  // CHECK: %[[CAST_BS:.+]] = index_cast %[[BS]] : index to i32
  // CHECK: hal.command_buffer.push_constants %[[UNUSED0:.+]], %[[UNUSED1:.+]], offset = 0, values = [%[[CAST_BS]]] : i32
  // Note that multiple dispatches in the stream verifies that transient
  // allocation is covering all ops.
  %0 = flow.ex.stream.fragment(%arg1 = %cst : index, %arg2 = %arg0 : tensor<?x128xf32>, %arg3 = %bs : index) -> tensor<?x128xf32> {
    %1 = shapex.make_ranked_shape %arg3 : (index) -> !shapex.ranked_shape<[?,128]>
    %2 = shapex.tie_shape %arg2, %1 : tensor<?x128xf32>, !shapex.ranked_shape<[?,128]>
    %3 = flow.dispatch @ex0::@entry0[%arg1] (%2, %arg3) : (tensor<?x128xf32>, index) -> tensor<?x128xf32>
    %4 = shapex.tie_shape %3, %1 : tensor<?x128xf32>, !shapex.ranked_shape<[?,128]>
    %5 = flow.dispatch @ex0::@entry0[%arg1] (%4, %arg3) : (tensor<?x128xf32>, index) -> tensor<?x128xf32>
    %6 = shapex.tie_shape %5, %1 : tensor<?x128xf32>, !shapex.ranked_shape<[?,128]>
    %7 = flow.dispatch @ex0::@entry0[%arg1] (%6, %arg3) : (tensor<?x128xf32>, index) -> tensor<?x128xf32>
    %8 = shapex.tie_shape %7, %1 : tensor<?x128xf32>, !shapex.ranked_shape<[?,128]>
    flow.return %8 : tensor<?x128xf32>
  }
  return %0 : tensor<?x128xf32>
}

// -----

hal.executable @ex attributes {sym_visibility = "private"} {
  hal.interface @legacy_io {
    hal.interface.binding @arg0, set=0, binding=0, type="StorageBuffer", access="Read"
    hal.interface.binding @ret0, set=0, binding=1, type="StorageBuffer", access="Write|Discard"
  }
  hal.executable.target @tgt, filter="dylib-llvm-aot" {
    hal.executable.entry_point @entry attributes {
      interface = @legacy_io,
      ordinal = 0 : i32,
      signature = (!flow.dispatch.input<7x4x24xf32>, !flow.dispatch.output<4x7x1024xf32>) -> ()
    }
    module {}
  }
}

// CHECK-LABEL: func @static_tiled_dispatch
func @static_tiled_dispatch(%arg0: tensor<7x4x24xf32>) -> tensor<4x7x1024xf32> {
  %c1024 = constant 1024 : index
  %c512 = constant 512 : index
  // CHECK: %[[CMD:.+]] = hal.command_buffer.create {{.+}}, OneShot, "Transfer|Dispatch"
  // CHECK-NEXT: hal.command_buffer.begin %[[CMD]]
  %1 = flow.ex.stream.fragment(
      %arg3 = %arg0 : tensor<7x4x24xf32>,
      %arg6 = %c1024 : index,
      %arg7 = %c512 : index
    ) -> tensor<4x7x1024xf32> {
    // CHECK: hal.command_buffer.push_descriptor_set %[[CMD]], %executable_layout, set = %c0, bindings = [%c0 = (%arg0, %c0, %c2688), %c1 = (%buffer, %c0, %c114688)]
    // CHECK: hal.command_buffer.dispatch.symbol {{.+}}, @ex::@tgt::@entry, workgroup_xyz
    %0 = flow.dispatch @ex::@entry[%arg6, %arg7, %arg7] (%arg3) : (tensor<7x4x24xf32>) -> tensor<4x7x1024xf32>
    flow.return %0 : tensor<4x7x1024xf32>
  }
  // CHECK: hal.command_buffer.end %[[CMD]]
  return %1 : tensor<4x7x1024xf32>
}

// -----

hal.executable @ex attributes {sym_visibility = "private"} {
  hal.interface @legacy_io attributes {push_constants = 4 : i32} {
    hal.interface.binding @arg0, set=0, binding=0, type="StorageBuffer", access="Read"
    hal.interface.binding @ret0, set=0, binding=1, type="StorageBuffer", access="Write|Discard"
  }
  hal.executable.target @tgt, filter="dylib-llvm-aot" {
    hal.executable.entry_point @entry attributes {
      interface = @legacy_io,
      ordinal = 0 : i32,
      signature = (!flow.dispatch.input<7x?x24x?xf32>, !flow.dispatch.output<?x?x1024xf32>, index, index, index, index) -> ()
    }
    module {}
  }
}

// CHECK-LABEL: func @dynamic_tiled_dispatch
func @dynamic_tiled_dispatch(%arg0: tensor<7x?x24x?xf32>, %arg1: index, %arg2: index) -> tensor<?x?x1024xf32> {
  %c1024 = constant 1024 : index
  %c512 = constant 512 : index
  // CHECK: %[[CMD:.+]] = hal.command_buffer.create {{.+}}, OneShot, "Transfer|Dispatch"
  // CHECK-NEXT: hal.command_buffer.begin %[[CMD]]
  %2 = flow.ex.stream.fragment(
      %arg3 = %arg0 : tensor<7x?x24x?xf32>,
      %arg4 = %arg1 : index,
      %arg5 = %arg2 : index,
      %arg6 = %c1024 : index,
      %arg7 = %c512 : index
    ) -> tensor<?x?x1024xf32> {
    %3 = shapex.make_ranked_shape %arg4, %arg5 : (index, index) -> !shapex.ranked_shape<[7,?,24,?]>
    %4 = shapex.make_ranked_shape %arg5, %arg4 : (index, index) -> !shapex.ranked_shape<[?,?,1024]>
    %5 = shapex.tie_shape %arg3, %3 : tensor<7x?x24x?xf32>, !shapex.ranked_shape<[7,?,24,?]>
    // CHECK: hal.command_buffer.push_constants %[[CMD]], %executable_layout, offset = 0, values = [%{{.+}}, %{{.+}}, %{{.+}}, %{{.+}}] : i32
    // CHECK: hal.command_buffer.push_descriptor_set %[[CMD]], %executable_layout, set = %c0, bindings = [%c0 = (%arg0, %c0, %9), %c1 = (%buffer, %c0, %12)]

    // CHECK: #hal.device.match.id<"dylib*">(
    // CHECK-SAME: %[[CMD_INNER:.+]] = %cmd : !hal.command_buffer,
    // CHECK-SAME: %[[COUNT_X_INNER:.+]] = %c1024 : index,
    // CHECK-SAME: %[[COUNT_Y_INNER:.+]] = %c512 : index,
    // CHECK-SAME: %[[COUNT_Z_INNER:.+]] = %c512 : index

    // This makes me so sad.
    //      CHECK: %[[C1:.+]] = constant 1
    // CHECK-NEXT: %[[COUNT_X_TMP:.+]] = addi %[[COUNT_X_INNER]], %[[C1]]
    // CHECK-NEXT: %[[COUNT_X:.+]] = subi %[[COUNT_X_TMP]], %[[C1]]
    // CHECK-NEXT: %[[COUNT_Y_TMP:.+]] = addi %[[COUNT_Y_INNER]], %[[C1]]
    // CHECK-NEXT: %[[COUNT_Y:.+]] = subi %[[COUNT_Y_TMP]], %[[C1]]
    // CHECK-NEXT: %[[COUNT_Z_TMP:.+]] = addi %[[COUNT_Z_INNER]], %[[C1]]
    // CHECK-NEXT: %[[COUNT_Z:.+]] = subi %[[COUNT_Z_TMP]], %[[C1]]

    // CHECK: hal.command_buffer.dispatch.symbol %[[CMD_INNER]], @ex::@tgt::@entry, workgroup_xyz =
    // CHECK-SAME: [%[[COUNT_X]], %[[COUNT_Y]], %[[COUNT_Z]]]
    %6 = flow.dispatch @ex::@entry[%arg6, %arg7, %arg7] (%5, %arg4, %arg5, %arg5, %arg4) : (tensor<7x?x24x?xf32>, index, index, index, index) -> tensor<?x?x1024xf32>
    %7 = shapex.tie_shape %6, %4 : tensor<?x?x1024xf32>, !shapex.ranked_shape<[?,?,1024]>
    flow.return %7 : tensor<?x?x1024xf32>
  }
  // CHECK: hal.command_buffer.end %[[CMD]]
  return %2 : tensor<?x?x1024xf32>
}

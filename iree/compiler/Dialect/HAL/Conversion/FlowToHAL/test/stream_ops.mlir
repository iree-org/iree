// RUN: iree-opt -split-input-file -iree-convert-to-hal -canonicalize %s | IreeFileCheck %s

hal.executable @ex0 {
  hal.interface @interface {
    hal.interface.binding @s0b0, set=0, binding=0, type="StorageBuffer", access="Read"
    hal.interface.binding @s0b1, set=0, binding=1, type="StorageBuffer", access="Read|Write"
  }
  hal.executable.target @vmla, filter="vmla" {
    hal.executable.entry_point @entry0 attributes {
      interface = @interface,
      ordinal = 0 : index,
      signature = (tensor<128xf32>) -> tensor<128xf32>
    }
    module {}
  }
}

// CHECK-LABEL: func @multipleDispatches
// CHECK-SAME: %[[INPUT_BUF:.+]]: !hal.buffer
func @multipleDispatches(%input: tensor<128xf32>) -> tensor<128xf32> {
  // CHECK-DAG: %[[C0:.+]] = constant 0
  // CHECK-DAG: %[[C128:.+]] = constant 128
  %cst = constant 128 : index
  //      CHECK: %[[RET_BUF:.+]] = hal.allocator.allocate
  // CHECK-SAME:   type("HostVisible|DeviceVisible|DeviceLocal")
  // CHECK-SAME:   usage("Transfer|Mapping|Dispatch")
  // CHECK-SAME:   : !hal.buffer{%c512}
  //      CHECK: %[[TMP_BUF:.+]] = hal.allocator.allocate
  // CHECK-SAME:   type("DeviceVisible|DeviceLocal")
  // CHECK-SAME:   usage("Transfer|Dispatch")
  // CHECK-SAME:   : !hal.buffer{%c512}
  //      CHECK: %[[CMD:.+]] = hal.command_buffer.create
  // CHECK-SAME:   mode(OneShot)
  // CHECK-SAME:   categories("Transfer|Dispatch")
  // CHECK-NEXT: hal.command_buffer.begin<%[[CMD]]
  %0 = flow.ex.stream.fragment(%cst, %input) : (index, tensor<128xf32>) -> tensor<128xf32> =
      (%arg1: index, %arg2: tensor<128xf32>) -> tensor<128xf32> {
    //  CHECK-DAG: %[[EXE_LAYOUT:.+]] = hal.executable_layout.lookup
    //      CHECK: hal.command_buffer.push_descriptor_set
    // CHECK-SAME:   layout(%[[EXE_LAYOUT]] : !hal.executable_layout)[%c0]
    // CHECK-SAME:   bindings([
    // CHECK-NEXT:     %c0 = (%[[INPUT_BUF]] : !hal.buffer)[%c0, %c512],
    // CHECK-NEXT:     %c1 = (%[[TMP_BUF]] : !hal.buffer)[%c0, %c512]
    //      CHECK: hal.command_buffer.dispatch.symbol
    // CHECK-SAME:   target(@ex0::@vmla::@entry0)
    //      CHECK: hal.command_buffer.execution_barrier
    %1 = flow.dispatch @ex0::@entry0[%arg1](%arg2) : (tensor<128xf32>) -> tensor<128xf32>
    //      CHECK: hal.command_buffer.push_descriptor_set
    //      CHECK: hal.command_buffer.dispatch.symbol
    // CHECK-SAME:   target(@ex0::@vmla::@entry0)
    //      CHECK: hal.command_buffer.execution_barrier
    %2 = flow.dispatch @ex0::@entry0[%arg1](%1) : (tensor<128xf32>) -> tensor<128xf32>
    flow.return %2 : tensor<128xf32>
  }
  // CHECK: hal.command_buffer.end<%[[CMD]]
  // CHECK-NEXT: hal.ex.submit_and_wait {{.+}}, %[[CMD]]
  // CHECK-NEXT: return %[[RET_BUF]]
  return %0 : tensor<128xf32>
}

// -----

// CHECK-LABEL: @tensorSlice
// CHECK-SAME: (%[[SBUF:.+]]:{{.+}})
func @tensorSlice(%arg0 : tensor<5x24x48xf32>) -> tensor<3x24x48xf32> {
  %c0 = constant 0 : index
  %c2 = constant 2 : index
  %c3 = constant 3 : index
  %c24 = constant 24 : index
  %c48 = constant 48 : index
  // CHECK: %[[RET_BUF:.+]] = hal.allocator.allocate
  // CHECK: %[[CMD:.+]] = hal.command_buffer.create
  // CHECK-NEXT: hal.command_buffer.begin<%[[CMD]]
  %2 = flow.ex.stream.fragment(%arg0, %c0, %c2, %c3, %c24, %c48)
      : (tensor<5x24x48xf32>, index, index, index, index, index) -> tensor<3x24x48xf32> =
      (%arg2 : tensor<5x24x48xf32>, %arg3 : index, %arg4 : index, %arg5 : index,
       %arg6 : index, %arg7 : index) -> tensor<3x24x48xf32> {
     // CHECK-NEXT: hal.command_buffer.copy_buffer<%[[CMD]]
     // CHECK-SAME:   source(%[[SBUF]] : !hal.buffer)[%c9216]
     // CHECK-SAME:   target(%[[RET_BUF]] : !hal.buffer)[%c0]
     // CHECK-SAME:   length(%c13824)
     %slice = flow.tensor.slice %arg2[%arg4, %arg3, %arg3 for %arg5, %arg6, %arg7]
         : tensor<5x24x48xf32> -> tensor<3x24x48xf32>
     flow.return %slice : tensor<3x24x48xf32>
  }
  return %2 : tensor<3x24x48xf32>
}

// -----

// CHECK-LABEL: @tensorUpdate
// CHECK-SAME: (%[[UBUF:.+]]:{{.+}}, %[[TBUF:.+]]:{{.+}})
func @tensorUpdate(%arg0 : tensor<1x1x10xf32>, %arg1 : tensor<5x1x10xf32>) -> tensor<5x1x10xf32> {
  %c4 = constant 4 : index
  %c1 = constant 1 : index
  // CHECK: %[[RET_BUF:.+]] = hal.allocator.allocate
  // CHECK: %[[CMD:.+]] = hal.command_buffer.create
  // CHECK-NEXT: hal.command_buffer.begin<%[[CMD]]
  %0 = flow.ex.stream.fragment(%arg0, %arg1, %c4, %c1) : (tensor<1x1x10xf32>, tensor<5x1x10xf32>, index, index) -> tensor<5x1x10xf32> =
      (%arg2: tensor<1x1x10xf32>, %arg3: tensor<5x1x10xf32>, %arg4: index, %arg5: index) -> tensor<5x1x10xf32> {
    // CHECK-NEXT: hal.command_buffer.copy_buffer
    // CHECK-SAME:   source(%[[TBUF]] : !hal.buffer)[%c0]
    // CHECK-SAME:   target(%[[RET_BUF]] : !hal.buffer)[%c0]
    // CHECK-SAME:   length(%c200)
    // CHECK: hal.command_buffer.execution_barrier
    %clone = flow.tensor.clone %arg3 : tensor<5x1x10xf32>
    // CHECK-NEXT: hal.command_buffer.copy_buffer
    // CHECK-SAME:   source(%[[UBUF]] : !hal.buffer)[%c0]
    // CHECK-SAME:   target(%[[RET_BUF]] : !hal.buffer)[%c204]
    // CHECK-SAME:   length(%c40)
    %1 = flow.tensor.update %arg2, %clone[%arg4, %arg5, %arg5] : tensor<1x1x10xf32> -> tensor<5x1x10xf32>
    flow.return %1 : tensor<5x1x10xf32>
  }
  // CHECK: hal.command_buffer.end<%[[CMD]]
  // CHECK: return %[[RET_BUF]]
  return %0 : tensor<5x1x10xf32>
}

// -----

hal.executable @ex0 {
  hal.interface @interface attributes {push_constants = 2 : index} {
    hal.interface.binding @s0b0, set=0, binding=0, type="StorageBuffer", access="Read"
    hal.interface.binding @s0b1, set=0, binding=1, type="StorageBuffer", access="Read|Write"
  }
  hal.executable.target @vmla, filter="vmla" {
    hal.executable.entry_point @entry0 attributes {
      interface = @interface,
      ordinal = 0 : index,
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
  // CHECK: hal.command_buffer.push_constants
  // CHECK-SAME:   layout({{.+}} : !hal.executable_layout)
  // CHECK-SAME:   offset(0)
  // CHECK-SAME:   values([%[[CAST_BS]]]) : i32
  // Note that multiple dispatches in the stream verifies that transient
  // allocation is covering all ops.
  %0 = flow.ex.stream.fragment(%cst, %arg0, %bs) : (index, tensor<?x128xf32>{%cst}, index) -> tensor<?x128xf32>{%cst} =
      (%arg1: index, %arg2: tensor<?x128xf32>, %arg3: index) -> tensor<?x128xf32> {
    %3 = flow.dispatch @ex0::@entry0[%arg1](%arg2, %arg3) : (tensor<?x128xf32>{%arg3}, index) -> tensor<?x128xf32>{%arg3}
    %5 = flow.dispatch @ex0::@entry0[%arg1](%3, %arg3) : (tensor<?x128xf32>{%arg3}, index) -> tensor<?x128xf32>{%arg3}
    %7 = flow.dispatch @ex0::@entry0[%arg1](%5, %arg3) : (tensor<?x128xf32>{%arg3}, index) -> tensor<?x128xf32>{%arg3}
    flow.return %7 : tensor<?x128xf32>
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
      ordinal = 0 : index,
      signature = (!flow.dispatch.tensor<readonly:7x4x24xf32>, !flow.dispatch.tensor<writeonly:4x7x1024xf32>) -> ()
    }
    module {}
  }
}

// CHECK-LABEL: func @static_tiled_dispatch
// CHECK-SAME: %[[INPUT:.+]]: !hal.buffer
func @static_tiled_dispatch(%input: tensor<7x4x24xf32>) -> tensor<4x7x1024xf32> {
  %c1024 = constant 1024 : index
  %c512 = constant 512 : index
  // CHECK: %[[CMD:.+]] = hal.command_buffer.create
  // CHECK-NEXT: hal.command_buffer.begin<%[[CMD]]
  %1 = flow.ex.stream.fragment(%input, %c1024, %c512) : (tensor<7x4x24xf32>, index, index) -> tensor<4x7x1024xf32> =
      (%arg3: tensor<7x4x24xf32>, %arg6: index, %arg7: index) -> tensor<4x7x1024xf32> {
    //      CHECK: hal.command_buffer.push_descriptor_set
    // CHECK-SAME:   layout(%executable_layout : !hal.executable_layout)[%c0]
    // CHECK-SAME:   bindings([
    // CHECK-NEXT:     %c0 = (%[[INPUT]] : !hal.buffer)[%c0, %c2688],
    // CHECK-NEXT:     %c1 = (%{{.+}} : !hal.buffer)[%c0, %c114688]
    //      CHECK: hal.command_buffer.dispatch.symbol
    // CHECK-SAME:   target(@ex::@tgt::@entry)
    %0 = flow.dispatch @ex::@entry[%arg6, %arg7, %arg7](%arg3) : (tensor<7x4x24xf32>) -> tensor<4x7x1024xf32>
    flow.return %0 : tensor<4x7x1024xf32>
  }
  // CHECK: hal.command_buffer.end<%[[CMD]]
  return %1 : tensor<4x7x1024xf32>
}

// -----

hal.executable @ex attributes {sym_visibility = "private"} {
  hal.interface @legacy_io attributes {push_constants = 4 : index} {
    hal.interface.binding @arg0, set=0, binding=0, type="StorageBuffer", access="Read"
    hal.interface.binding @ret0, set=0, binding=1, type="StorageBuffer", access="Write|Discard"
  }
  hal.executable.target @tgt, filter="dylib-llvm-aot" {
    hal.executable.entry_point @entry attributes {
      interface = @legacy_io,
      ordinal = 0 : index,
      signature = (!flow.dispatch.tensor<readonly:7x?x24x?xf32>, !flow.dispatch.tensor<writeonly:?x?x1024xf32>, index, index, index, index) -> ()
    }
    module {}
  }
}

// CHECK-LABEL: func @dynamic_tiled_dispatch
// CHECK-SAME: %[[INPUT:.+]]: !hal.buffer
func @dynamic_tiled_dispatch(%arg0: tensor<7x?x24x?xf32>, %arg1: index, %arg2: index) -> tensor<?x?x1024xf32> {
  %c1024 = constant 1024 : index
  %c512 = constant 512 : index
  // CHECK: %[[CMD:.+]] = hal.command_buffer.create
  // CHECK-NEXT: hal.command_buffer.begin<%[[CMD]]
  %2 = flow.ex.stream.fragment(%arg0, %arg1, %arg2, %c1024, %c512) : (tensor<7x?x24x?xf32>{%arg1, %arg2}, index, index, index, index) -> tensor<?x?x1024xf32>{%arg2, %arg1} =
      (%arg3: tensor<7x?x24x?xf32>, %arg4: index, %arg5: index, %arg6: index, %arg7: index) -> tensor<?x?x1024xf32> {
    //      CHECK: hal.command_buffer.push_constants<%[[CMD]]
    // CHECK-SAME:   layout(%executable_layout
    // CHECK-SAME:   offset(0)
    // CHECK-SAME:   values([%{{.+}}, %{{.+}}, %{{.+}}, %{{.+}}]) : i32, i32, i32, i32
    //      CHECK: hal.command_buffer.push_descriptor_set<%[[CMD]]
    // CHECK-SAME:   layout(%executable_layout : !hal.executable_layout)[%c0]
    // CHECK-SAME:   bindings([
    // CHECK-NEXT:     %c0 = (%[[INPUT]] : !hal.buffer)[%c0, %9],
    // CHECK-NEXT:     %c1 = (%{{.+}} : !hal.buffer)[%c0, %12]

    // CHECK: #hal.device.match.id<"dylib*">(
    // CHECK-SAME: %[[CMD_INNER:.+]] = %cmd : !hal.command_buffer,
    // CHECK-SAME: %[[COUNT_X_INNER:.+]] = %c1024 : index,
    // CHECK-SAME: %[[COUNT_Y_INNER:.+]] = %c512 : index,
    // CHECK-SAME: %[[COUNT_Z_INNER:.+]] = %c512 : index

    // This makes me so sad.
    // If you are improving folding/canonicalization of these ops and come
    // across this feel free to remove it all. And let me know so I can buy you
    // a drink :)
    //      CHECK: %[[C1:.+]] = constant 1
    // CHECK-NEXT: %[[COUNT_X_TMP:.+]] = addi %[[COUNT_X_INNER]], %[[C1]]
    // CHECK-NEXT: %[[COUNT_X:.+]] = subi %[[COUNT_X_TMP]], %[[C1]]
    // CHECK-NEXT: %[[COUNT_Y_TMP:.+]] = addi %[[COUNT_Y_INNER]], %[[C1]]
    // CHECK-NEXT: %[[COUNT_Y:.+]] = subi %[[COUNT_Y_TMP]], %[[C1]]
    // CHECK-NEXT: %[[COUNT_Z_TMP:.+]] = addi %[[COUNT_Z_INNER]], %[[C1]]
    // CHECK-NEXT: %[[COUNT_Z:.+]] = subi %[[COUNT_Z_TMP]], %[[C1]]

    //      CHECK: hal.command_buffer.dispatch.symbol<%[[CMD_INNER]]
    // CHECK-SAME:   target(@ex::@tgt::@entry)
    // CHECK-SAME:   workgroups([%[[COUNT_X]], %[[COUNT_Y]], %[[COUNT_Z]]])
    %6 = flow.dispatch @ex::@entry[%arg6, %arg7, %arg7](%arg3, %arg4, %arg5, %arg5, %arg4) : (tensor<7x?x24x?xf32>{%arg4, %arg5}, index, index, index, index) -> tensor<?x?x1024xf32>{%arg5, %arg4}
    flow.return %6 : tensor<?x?x1024xf32>
  }
  // CHECK: hal.command_buffer.end<%[[CMD]]
  return %2 : tensor<?x?x1024xf32>
}

// -----

hal.executable @pad_dispatch_0 attributes {sym_visibility = "private"} {
  hal.interface @interface_io {
    hal.interface.binding @ro0, set=0, binding=0, type="StorageBuffer", access="Read"
    hal.interface.binding @wo1, set=0, binding=1, type="StorageBuffer", access="Write|Discard"
  }
  hal.executable.target @tgt, filter="dylib-llvm-aot" {
    hal.executable.entry_point @pad_dispatch_0 attributes {
      interface = @interface_io,
      ordinal = 0 : index,
      signature = (!flow.dispatch.tensor<readonly:i32>, !flow.dispatch.tensor<writeonly:3x9xi32>) -> ()
    }
    module {}
  }
}

hal.executable @pad_dispatch_1 attributes {sym_visibility = "private"} {
  hal.interface @interface_io {
    hal.interface.binding @ro0, set=0, binding=0, type="StorageBuffer", access="Read"
    hal.interface.binding @rw1, set=0, binding=1, type="StorageBuffer", access="Read|Write"
  }
  hal.executable.target @tgt, filter="dylib-llvm-aot" {
    hal.executable.entry_point @pad_dispatch_1 attributes {
      interface = @interface_io,
      ordinal = 0 : index,
      signature = (!flow.dispatch.tensor<readonly:2x3xi32>, !flow.dispatch.tensor<readwrite:3x9xi32>) -> ()
    }
    module {}
  }
}

// CHECK-LABEL: func @dispatch_tied_buffer
// CHECK-SAME: (%[[FILL:.+]]: !hal.buffer, %[[INPUT:.+]]: !hal.buffer)
func @dispatch_tied_buffer(%fill: tensor<i32>, %input: tensor<2x3xi32>) -> tensor<3x9xi32> {
  //      CHECK: %[[OUTPUT:.+]] = hal.allocator.allocate
  // CHECK-SAME:   type("HostVisible|DeviceVisible|DeviceLocal")
  // CHECK-SAME:   usage("Transfer|Mapping|Dispatch")
  %0 = flow.ex.stream.fragment(%fill, %input) : (tensor<i32>, tensor<2x3xi32>) -> tensor<3x9xi32> =
      (%arg0: tensor<i32>, %arg1: tensor<2x3xi32>) -> tensor<3x9xi32> {
    %c9 = constant 9 : index
    %c3 = constant 3 : index
    %c1 = constant 1 : index
    //      CHECK: %[[LAYOUT0:.+]] = hal.executable_layout.lookup
    // CHECK-SAME:   layouts([
    // CHECK-SAME:     #hal.descriptor_set_layout_binding<0, "StorageBuffer", R>,
    // CHECK-SAME:     #hal.descriptor_set_layout_binding<1, "StorageBuffer", DW>
    //      CHECK: hal.command_buffer.push_descriptor_set
    // CHECK-SAME:   layout(%[[LAYOUT0]] : !hal.executable_layout)[%{{.+}}]
    // CHECK-SAME:   bindings([
    // CHECK-NEXT:     %c0 = (%[[FILL]] : !hal.buffer)[%c0, %c4],
    // CHECK-NEXT:     %c1 = (%[[OUTPUT]] : !hal.buffer)[%c0, %c108]
    %3 = flow.dispatch @pad_dispatch_0::@pad_dispatch_0[%c9, %c3, %c1](%arg0) : (tensor<i32>) -> tensor<3x9xi32>
    //      CHECK: %[[LAYOUT1:.+]] = hal.executable_layout.lookup
    // CHECK-SAME:   layouts([
    // CHECK-SAME:     #hal.descriptor_set_layout_binding<0, "StorageBuffer", R>,
    // CHECK-SAME:     #hal.descriptor_set_layout_binding<1, "StorageBuffer", RW>
    //      CHECK: hal.command_buffer.push_descriptor_set
    // CHECK-SAME:   layout(%[[LAYOUT1]] : !hal.executable_layout)[%{{.+}}]
    // CHECK-SAME:   bindings([
    // CHECK-NEXT:     %c0 = (%[[INPUT]] : !hal.buffer)[%c0, %c24],
    // CHECK-NEXT:     %c1 = (%[[OUTPUT]] : !hal.buffer)[%c0, %c108]
    %4 = flow.dispatch @pad_dispatch_1::@pad_dispatch_1[%c9, %c3, %c1](%arg1, %3) : (tensor<2x3xi32>, tensor<3x9xi32>) -> %3
    flow.return %4 : tensor<3x9xi32>
  }
  return %0 : tensor<3x9xi32>
}

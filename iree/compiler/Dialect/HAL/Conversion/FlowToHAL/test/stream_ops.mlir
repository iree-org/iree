// RUN: iree-opt -split-input-file -iree-convert-flow-to-hal -canonicalize %s | IreeFileCheck %s

hal.executable @ex0 {
  hal.executable.entry_point @entry0 attributes {
    ordinal = 0 : i32,
    workgroup_size = dense<[32, 1, 1]> : vector<3xi32>
  }
}

// CHECK-LABEL: func @multipleDispatches
func @multipleDispatches(%arg0: tensor<128xf32>) -> tensor<128xf32> {
  // CHECK-DAG: [[C1:%.+]] = constant 1
  // CHECK-DAG: [[C4:%.+]] = constant 4
  // CHECK-DAG: [[C128:%.+]] = constant 128
  %cst = constant dense<[128, 1, 1]> : vector<3xi32>
  // CHECK: [[RET_BUF:%.+]] = hal.allocator.allocate {{.+}}, "HostVisible|DeviceVisible|DeviceLocal", "Constant|Transfer|Mapping|Dispatch"
  // CHECK-NEXT: hal.ex.defer_release [[RET_BUF]]
  // CHECK: [[TMP_BUF:%.+]] = hal.allocator.allocate {{.+}}, "DeviceVisible|DeviceLocal", "Transfer|Dispatch"
  // CHECK-NEXT: hal.ex.defer_release [[TMP_BUF]]
  // CHECK: [[CMD:%.+]] = hal.command_buffer.create {{.+}}, "OneShot", "Transfer|Dispatch"
  // CHECK-NEXT: hal.command_buffer.begin [[CMD]]
  %0 = flow.ex.stream.fragment(%arg1 = %cst : vector<3xi32>, %arg2 = %arg0 : tensor<128xf32>) -> tensor<128xf32> {
    // CHECK: [[EXE:%.+]] = hal.ex.cache_executable {{.+}}, @ex0 : !iree.ref<!hal.executable>
    // CHECK-NEXT: hal.ex.push_binding [[CMD]], 0, %arg0, shape = [
    // CHECK-SAME:   [[C128]]
    // CHECK-SAME: ], element_type = 50331680
    // CHECK-NEXT: hal.ex.defer_release
    // CHECK-NEXT: hal.ex.push_binding [[CMD]], 1, [[TMP_BUF]], shape = [
    // CHECK-SAME:   [[C128]]
    // CHECK-SAME: ], element_type = 50331680
    // CHECK-NEXT: hal.ex.defer_release
    // CHECK-NEXT: hal.command_buffer.dispatch [[CMD]], [[EXE]], entry_point = 0, workgroup_xyz = [
    // CHECK-SAME:   [[C4]], [[C1]], [[C1]]
    // CHECK-SAME: ]
    // CHECK: hal.command_buffer.execution_barrier
    %1 = flow.dispatch @ex0::@entry0[%arg1 : vector<3xi32>](%arg2) : (tensor<128xf32>) -> tensor<128xf32>
    // CHECK: hal.ex.push_binding [[CMD]], 0, [[TMP_BUF]], shape = [
    // CHECK-SAME:   [[C128]]
    // CHECK-SAME: ], element_type = 50331680
    // CHECK-NEXT: hal.ex.defer_release
    // CHECK-NEXT: hal.ex.push_binding [[CMD]], 1, [[RET_BUF]], shape = [
    // CHECK-SAME:   [[C128]]
    // CHECK-SAME: ], element_type = 50331680
    // CHECK-NEXT: hal.ex.defer_release
    // CHECK-NEXT: hal.command_buffer.dispatch [[CMD]], {{.+}}, entry_point = 0, workgroup_xyz = [
    // CHECK-SAME:   [[C4]], [[C1]], [[C1]]
    // CHECK-SAME: ]
    // CHECK: hal.command_buffer.execution_barrier
    %2 = flow.dispatch @ex0::@entry0[%arg1 : vector<3xi32>](%1) : (tensor<128xf32>) -> tensor<128xf32>
    flow.return %2 : tensor<128xf32>
  }
  // CHECK: hal.command_buffer.end [[CMD]]
  // CHECK-NEXT: hal.ex.submit_and_wait {{.+}}, [[CMD]]
  // CHECK-NEXT: return [[RET_BUF]]
  return %0 : tensor<128xf32>
}

// -----

// CHECK-LABEL: @tensorUpdate
// CHECK-SAME: ([[UBUF:%.+]]:{{.+}}, [[TBUF:%.+]]:{{.+}})
func @tensorUpdate(%arg0 : tensor<1x1x10xf32>, %arg1 : tensor<5x1x10xf32>) -> tensor<5x1x10xf32> {
  // CHECK: [[C0:%.+]] = constant 0
  %c4 = constant 4 : i32
  %c1 = constant 1 : i32
  // CHECK: [[RET_BUF:%.+]] = hal.allocator.allocate
  // CHECK: [[CMD:%.+]] = hal.command_buffer.create
  // CHECK-NEXT: hal.command_buffer.begin [[CMD]]
  %0 = flow.ex.stream.fragment(%arg2 = %arg0 : tensor<1x1x10xf32>, %arg3 = %arg1 : tensor<5x1x10xf32>, %arg4 = %c4 : i32, %arg5 = %c1 : i32) -> tensor<5x1x10xf32> {
    // CHECK: [[UOFF:%.+]], [[ULEN:%.+]] = hal.allocator.compute_range {{%.+}}
    // CHECK: [[TLEN:%.+]] = hal.allocator.compute_size {{%.+}}
    // CHECK-NEXT: hal.command_buffer.copy_buffer [[CMD]], [[TBUF]], [[C0]], [[RET_BUF]], [[C0]], [[TLEN]]
    // CHECK: hal.command_buffer.execution_barrier
    // CHECK-NEXT: hal.command_buffer.copy_buffer [[CMD]], [[UBUF]], [[C0]], [[RET_BUF]], [[UOFF]], [[ULEN]]
    %1 = flow.tensor.update %arg2, %arg3[%arg4, %arg5, %arg5] : tensor<1x1x10xf32> -> tensor<5x1x10xf32>
    flow.return %1 : tensor<5x1x10xf32>
  }
  // CHECK: hal.command_buffer.end [[CMD]]
  // CHECK: return [[RET_BUF]]
  return %0 : tensor<5x1x10xf32>
}

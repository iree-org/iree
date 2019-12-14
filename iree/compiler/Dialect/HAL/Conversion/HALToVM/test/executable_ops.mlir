// RUN: iree-opt -split-input-file -iree-convert-hal-to-vm %s | IreeFileCheck %s

hal.executable @exe {
  hal.executable.entry_point @entry attributes {ordinal = 0 : i32, workgroup_size = dense<[32, 1, 1]> : vector<3xi32>}
  hal.executable.entry_point @entry_alias attributes {ordinal = 0 : i32, workgroup_size = dense<[32, 1, 1]> : vector<3xi32>}
  hal.executable.binary attributes {data = dense<[0, 1, 2, 3]> : vector<4xi8>, format = 1230128453 : i32}
  hal.executable.binary attributes {data = dense<[4, 5, 6, 7]> : vector<4xi8>, format = 1230128454 : i32}
}

// CHECK-DAG: vm.rodata @exe_data_11124442505 dense<[0, 1, 2, 3]> : vector<4xi8>
// CHECK-DAG: vm.rodata @exe_data_11124442506 dense<[4, 5, 6, 7]> : vector<4xi8>
// CHECK-DAG: vm.global.ref @exe_cached mutable : !ireex.ref<!hal.executable>
// CHECK-DAG: vm.func @exe(%arg0: !ireex.ref<!hal.device>) -> !ireex.ref<!hal.executable> {
// CHECK:        %exe_cached = vm.global.load.ref @exe_cached : !ireex.ref<!hal.executable>
// CHECK-NEXT:   %rnz = vm.cmp.nz.ref %exe_cached : !ireex.ref<!hal.executable>
// CHECK-NEXT:   vm.cond_br %rnz, ^bb1(%exe_cached : !ireex.ref<!hal.executable>), ^bb2
// CHECK-NEXT: ^bb1(%0: !ireex.ref<!hal.executable>):
// CHECK-NEXT:   vm.return %0 : !ireex.ref<!hal.executable>
// CHECK-NEXT: ^bb2:
// CHECK-NEXT:   %c1230128453 = vm.const.i32 1230128453 : i32
// CHECK-NEXT:   %c1230128454 = vm.const.i32 1230128454 : i32
// CHECK-NEXT:   %1 = vm.call @hal.ex.match_supported_executable_format(%arg0, %c1230128453, %c1230128454) : (!ireex.ref<!hal.device>, i32, i32) -> i32
// CHECK-NEXT:   vm.br ^bb3(%1 : i32)
// CHECK-NEXT: ^bb3(%2: i32):
// CHECK-NEXT:   %c1230128453_0 = vm.const.i32 1230128453 : i32
// CHECK-NEXT:   %eq = vm.cmp.eq.i32 %2, %c1230128453_0 : i32
// CHECK-NEXT:   vm.cond_br %eq, ^bb4(%2 : i32), ^bb5(%2 : i32)
// CHECK-NEXT: ^bb4(%3: i32):
// CHECK-NEXT:   %exe_data_11124442505 = vm.const.ref.rodata @exe_data_11124442505 : !ireex.byte_buffer_ref
// CHECK-NEXT:   %ref = vm.call @hal.ex.cache_executable(%arg0, %3, %exe_data_11124442505) : (!ireex.ref<!hal.device>, i32, !ireex.byte_buffer_ref) -> !ireex.ref<!hal.executable>
// CHECK-NEXT:   vm.br ^bb7(%ref : !ireex.ref<!hal.executable>)
// CHECK-NEXT: ^bb5(%4: i32):
// CHECK-NEXT:   %c1230128454_1 = vm.const.i32 1230128454 : i32
// CHECK-NEXT:   %eq_2 = vm.cmp.eq.i32 %4, %c1230128454_1 : i32
// CHECK-NEXT:   vm.cond_br %eq_2, ^bb6(%4 : i32), ^bb8
// CHECK-NEXT: ^bb6(%5: i32):
// CHECK-NEXT:   %exe_data_11124442506 = vm.const.ref.rodata @exe_data_11124442506 : !ireex.byte_buffer_ref
// CHECK-NEXT:   %ref_3 = vm.call @hal.ex.cache_executable(%arg0, %5, %exe_data_11124442506) : (!ireex.ref<!hal.device>, i32, !ireex.byte_buffer_ref) -> !ireex.ref<!hal.executable>
// CHECK-NEXT:   vm.br ^bb7(%ref_3 : !ireex.ref<!hal.executable>)
// CHECK-NEXT: ^bb7(%6: !ireex.ref<!hal.executable>):
// CHECK-NEXT:   vm.global.store.ref @exe_cached, %6 : !ireex.ref<!hal.executable>
// CHECK-NEXT:   vm.return %6 : !ireex.ref<!hal.executable>
// CHECK-NEXT: ^bb8:
// CHECK-NEXT:   %null = vm.const.ref.zero : !ireex.ref<!hal.executable>
// CHECK-NEXT:   vm.return %null : !ireex.ref<!hal.executable>
// CHECK-NEXT: }

// -----

// CHECK-LABEL: @exeLookup
func @exeLookup(%arg0 : !ireex.ref<!hal.device>) -> !ireex.ref<!hal.executable> {
  // CHECK: vm.call @exe(%arg0) : (!ireex.ref<!hal.device>) -> !ireex.ref<!hal.executable>
  %0 = hal.ex.cache_executable %arg0, @exe : !ireex.ref<!hal.executable>
  return %0 : !ireex.ref<!hal.executable>
}

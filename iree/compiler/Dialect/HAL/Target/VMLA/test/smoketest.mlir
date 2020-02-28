// RUN: iree-opt -split-input-file -iree-hal-transformation-pipeline -iree-hal-target-backends=vmla %s | IreeFileCheck %s

flow.executable @simpleMath_ex_dispatch_0 {
  flow.dispatch.entry @simpleMath_rgn_dispatch_0 attributes {
    workload = dense<[4, 1, 1]> : vector<3xi32>
  }
  module {
    func @simpleMath_rgn_dispatch_0(%arg0: tensor<4xf32>) -> tensor<4xf32> {
      %0 = xla_hlo.add %arg0, %arg0 : tensor<4xf32>
      return %arg0 : tensor<4xf32>
    }
  }
}

// CHECK-LABEL: hal.executable @simpleMath_ex_dispatch_0 {
//  CHECK-NEXT:   hal.interface @legacy_io {
//  CHECK-NEXT:     hal.interface.binding @arg0, set=0, binding=0, type="StorageBuffer", access="Read"
//  CHECK-NEXT:     hal.interface.binding @ret0, set=0, binding=1, type="StorageBuffer", access="Write|Discard"
//  CHECK-NEXT:   }
//  CHECK-NEXT:   hal.executable.entry_point @simpleMath_rgn_dispatch_0 attributes {interface = @legacy_io, ordinal = 0 : i32, signature = (tensor<4xf32>) -> tensor<4xf32>, workgroup_size = dense<1> : vector<3xi32>}
//  CHECK-NEXT:   hal.executable.binary attributes {
//  CHECK-SAME:       data = dense<
//  CHECK-SAME:       format = 1447906369 : i32} {
//  CHECK-NEXT:     vm.module @module {
//  CHECK-NEXT:       vm.func @simpleMath_rgn_dispatch_0(%arg0: !vm.ref<!vmla.interface>) attributes {ordinal = 0 : i32} {
//  CHECK-NEXT:         %zero = vm.const.i32.zero : i32
//  CHECK-NEXT:         %c16 = vm.const.i32 16 : i32
//  CHECK-NEXT:         %c1 = vm.const.i32 1 : i32
//  CHECK-NEXT:         %ref = vm.call @vmla.interface.binding(%arg0, %zero, %zero) : (!vm.ref<!vmla.interface>, i32, i32) -> !vm.ref<!vmla.buffer>
//  CHECK-NEXT:         %ref_0 = vm.call @vmla.buffer.view(%ref, %zero, %c16) : (!vm.ref<!vmla.buffer>, i32, i32) -> !vm.ref<!vmla.buffer>
//  CHECK-NEXT:         %ref_1 = vm.call @vmla.interface.binding(%arg0, %zero, %c1) : (!vm.ref<!vmla.interface>, i32, i32) -> !vm.ref<!vmla.buffer>
//  CHECK-NEXT:         vm.call @vmla.buffer.copy(%ref_0, %zero, %ref_1, %zero, %c16) : (!vm.ref<!vmla.buffer>, i32, !vm.ref<!vmla.buffer>, i32, i32) -> ()
//  CHECK-NEXT:         vm.return
//  CHECK-NEXT:       }
//  CHECK-NEXT:       vm.export @simpleMath_rgn_dispatch_0 attributes {ordinal = 0 : i32}
//  CHECK-NEXT:       vm.import @vmla.buffer.view(%src : !vm.ref<!vmla.buffer>, %byte_offset : i32, %byte_length : i32) -> !vm.ref<!vmla.buffer> attributes {nosideeffects, ordinal = 0 : i32, sym_visibility = "private"}
//  CHECK-NEXT:       vm.import @vmla.buffer.copy(%src : !vm.ref<!vmla.buffer>, %src_byte_offset : i32, %dst : !vm.ref<!vmla.buffer>, %dst_byte_offset : i32, %byte_length : i32) attributes {ordinal = 1 : i32, sym_visibility = "private"}
//  CHECK-NEXT:       vm.import @vmla.interface.binding(%interface : !vm.ref<!vmla.interface>, %set : i32, %binding : i32) -> !vm.ref<!vmla.buffer> attributes {ordinal = 2 : i32, sym_visibility = "private"}
//  CHECK-NEXT:     }
//  CHECK-NEXT:   }
//  CHECK-NEXT: }

// -----

// TODO(benvanik): vmla reduction.
// flow.executable @reduction_ex_reduce_0_dim_0 {
//   flow.reduction.entry @reduction_rgn_reduce_0_dim_0_entry apply(@reduction_rgn_reduce_0_dim_0) attributes {
//     dimension = 1 : i32,
//     workgroup_size = dense<[32, 1, 1]> : vector<3xi32>,
//     workload = dense<[4, 1, 1]> : vector<3xi32>
//   }
//   module {
//     func @reduction_rgn_reduce_0_dim_0_entry(tensor<4x8xf32>, tensor<f32>) -> tensor<4xf32>
//     func @reduction_rgn_reduce_0_dim_0(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
//       %0 = xla_hlo.add %arg0, %arg1 : tensor<f32>
//       return %0 : tensor<f32>
//     }
//   }
// }

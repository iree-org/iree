// RUN: iree-compile --split-input-file --iree-hal-target-backends=vmvx \
// RUN:   --output-format=vm-bytecode \
// RUN:   --iree-vm-bytecode-module-output-format=flatbuffer-text %s \
// RUN:   --mlir-print-ir-after=iree-vm-ordinal-allocation 2>&1 | FileCheck %s

// This file has a few test programs that show how to mix `flow` dispatches into
// those created by the `linalg` dispatch region formation: the idea is to use
// any normal IREE input (mhlo/tosa/linalg/etc) on tensors and then also include
// `flow.dispatch` ops calling `stream.executable`s. `flow.executable`s could be
// used too but currently have some ergonomics issues that need to be resolved;
// the improved version of `flow.dispatch` (and `flow.dispatch.workgroups`) will
// be made part of the public `iree` dialect at which time this file will change
// to using that. The `flow`/`stream` dialects are generally not considered
// stable.

// A simple element-wise multiply of two static tensors:
//   %ret0 = %arg0 * %arg1
//
// The host code performs the dispatch with a workload of 4x1x1 - how many
// workgroups that gets distributed across is left to the HAL backend to decide
// based on the target device and how the work is tiled.
//
// The device code in the stream.executable is tiled - but does not need to be:
// the only thing we care about at this level is the bindings and any operands
// that may need to be passed from host->device.

// CHECK-LABEL: vm.module public @e2e
module @e2e {
// CHECK: vm.rodata private @executable_0_vmvx_bytecode_fb
stream.executable private @executable_0 {
  stream.executable.export public @dispatch workgroups(%arg0: index) -> (index, index, index) {
    %x, %y, %z = flow.dispatch.workgroup_count_from_dag_root %arg0
    stream.return %x, %y, %z : index, index, index
  }
  builtin.module {
    func.func @dispatch(%arg0: !stream.binding, %arg1: !stream.binding, %ret0: !stream.binding) {
      %c0 = arith.constant 0 : index
      %0 = stream.binding.subspan %arg0[%c0] : !stream.binding -> !flow.dispatch.tensor<readonly:tensor<4xf32>>
      %1 = stream.binding.subspan %arg1[%c0] : !stream.binding -> !flow.dispatch.tensor<readonly:tensor<4xf32>>
      %2 = stream.binding.subspan %ret0[%c0] : !stream.binding -> !flow.dispatch.tensor<writeonly:tensor<4xf32>>
      %3 = flow.dispatch.tensor.load %0, offsets = [0], sizes = [4], strides = [1] : !flow.dispatch.tensor<readonly:tensor<4xf32>> -> tensor<4xf32>
      %4 = flow.dispatch.tensor.load %1, offsets = [0], sizes = [4], strides = [1] : !flow.dispatch.tensor<readonly:tensor<4xf32>> -> tensor<4xf32>
      %5 = tensor.empty() : tensor<4xf32>
      %6 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%3, %4 : tensor<4xf32>, tensor<4xf32>) outs(%5 : tensor<4xf32>) attrs =  {name = "mul.1"} {
        ^bb0(%arg4: f32, %arg5: f32, %arg6: f32):
          %10 = arith.mulf %arg4, %arg5 : f32
          linalg.yield %10 : f32
        } -> tensor<4xf32>
      flow.dispatch.tensor.store %6, %2, offsets = [0], sizes = [4], strides = [1] : tensor<4xf32> -> !flow.dispatch.tensor<writeonly:tensor<4xf32>>
      return
    }
  }
}
// CHECK: vm.func private @__simple_mul_memoize_apply
// CHECK:   vm.call.variadic @hal.command_buffer.dispatch
func.func @simple_mul(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) -> tensor<4xf32> {
  %c4 = arith.constant 4 : index
  %ret0 = flow.dispatch @executable_0::@dispatch[%c4](%arg0, %arg1) : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
  return %ret0 : tensor<4xf32>
}
}  // module

// -----

// The same element-wise multiply but now in-place:
//   %arg0 = %arg0 * %arg1
//
// In-place operations can often introduce false dependencies between dispatches
// and should be avoided at this level in most cases - there's currently no cost
// model for making dispatches into in-place operations but it's something that
// would happen in the stream dialect after scheduling: two dispatches known to
// not be running concurrently and operating on the same resources could be made
// in-place.

// CHECK-LABEL: vm.module public @inplace
module @inplace {
// CHECK: vm.rodata private @executable_1_vmvx_bytecode_fb
stream.executable private @executable_1 {
  stream.executable.export public @dispatch workgroups(%arg0: index) -> (index, index, index) {
    %x, %y, %z = flow.dispatch.workgroup_count_from_dag_root %arg0
    stream.return %x, %y, %z : index, index, index
  }
  builtin.module {
    func.func @dispatch(%arg0: !stream.binding, %arg1: !stream.binding) {
      %c0 = arith.constant 0 : index
      %0 = stream.binding.subspan %arg0[%c0] : !stream.binding -> !flow.dispatch.tensor<readwrite:tensor<4xf32>>
      %1 = stream.binding.subspan %arg1[%c0] : !stream.binding -> !flow.dispatch.tensor<readonly:tensor<4xf32>>
      %3 = flow.dispatch.tensor.load %0, offsets = [0], sizes = [4], strides = [1] : !flow.dispatch.tensor<readwrite:tensor<4xf32>> -> tensor<4xf32>
      %4 = flow.dispatch.tensor.load %1, offsets = [0], sizes = [4], strides = [1] : !flow.dispatch.tensor<readonly:tensor<4xf32>> -> tensor<4xf32>
      %5 = tensor.empty() : tensor<4xf32>
      %6 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%3, %4 : tensor<4xf32>, tensor<4xf32>) outs(%5 : tensor<4xf32>) attrs =  {name = "mul.1"} {
        ^bb0(%arg4: f32, %arg5: f32, %arg6: f32):
          %10 = arith.mulf %arg4, %arg5 : f32
          linalg.yield %10 : f32
        } -> tensor<4xf32>
      flow.dispatch.tensor.store %6, %0, offsets = [0], sizes = [4], strides = [1] : tensor<4xf32> -> !flow.dispatch.tensor<readwrite:tensor<4xf32>>
      return
    }
  }
}
// CHECK: vm.func private @__simple_mul_inplace_memoize_apply
// CHECK:   vm.call.variadic @hal.command_buffer.dispatch
func.func @simple_mul_inplace(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) -> tensor<4xf32> {
  %c4 = arith.constant 4 : index
  %ret0 = flow.dispatch @executable_1::@dispatch[%c4](%arg0, %arg1) : (tensor<4xf32>, tensor<4xf32>) -> %arg0
  return %ret0 : tensor<4xf32>
}
}  // module

// -----

// The same element-wise multiply but now with dynamic shapes:
//   %ret0 = %arg0 * %arg1
//
// This shows how the shape dimensions are captured by the dispatch so that the
// host knows the shapes of the tensors and how the dimensions are passed as
// operands to the executable for association. Once we perform the host/device
// split the association allows tensor.dim ops in the device code to query the
// dynamic dimensions without needing to insert new host -> device transfers.
// Note that because of this explicit association the order of the dispatch
// operands doesn't matter as walking the SSA use-def chain up to the
// stream.binding.subspan allows them to be resolved directly.

// CHECK-LABEL: vm.module public @dynamic
module @dynamic {
// CHECK: vm.rodata private @executable_2_vmvx_bytecode_fb
stream.executable private @executable_2 {
  stream.executable.export public @dispatch workgroups(%arg0: index) -> (index, index, index) {
    %x, %y, %z = flow.dispatch.workgroup_count_from_dag_root %arg0
    stream.return %x, %y, %z : index, index, index
  }
  builtin.module {
    func.func @dispatch(%arg0: !stream.binding, %arg0_dim0: index, %arg1: !stream.binding, %arg1_dim0: index, %ret0: !stream.binding) {
      %c0 = arith.constant 0 : index
      %0 = stream.binding.subspan %arg0[%c0] : !stream.binding -> !flow.dispatch.tensor<readonly:tensor<?xf32>>{%arg0_dim0}
      %1 = stream.binding.subspan %arg1[%c0] : !stream.binding -> !flow.dispatch.tensor<readonly:tensor<?xf32>>{%arg1_dim0}
      %2 = stream.binding.subspan %ret0[%c0] : !stream.binding -> !flow.dispatch.tensor<writeonly:tensor<?xf32>>{%arg0_dim0}
      %3 = flow.dispatch.tensor.load %0, offsets = [0], sizes = [%arg0_dim0], strides = [1] : !flow.dispatch.tensor<readonly:tensor<?xf32>>{%arg0_dim0} -> tensor<?xf32>
      %4 = flow.dispatch.tensor.load %1, offsets = [0], sizes = [%arg1_dim0], strides = [1] : !flow.dispatch.tensor<readonly:tensor<?xf32>>{%arg1_dim0} -> tensor<?xf32>
      %5 = tensor.empty(%arg0_dim0) : tensor<?xf32>
      %6 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%3, %4 : tensor<?xf32>, tensor<?xf32>) outs(%5 : tensor<?xf32>) attrs =  {name = "mul.1"} {
        ^bb0(%arg6: f32, %arg7: f32, %arg8: f32):
          %10 = arith.mulf %arg6, %arg7 : f32
          linalg.yield %10 : f32
        } -> tensor<?xf32>
      flow.dispatch.tensor.store %6, %2, offsets = [0], sizes = [%arg0_dim0], strides = [1] : tensor<?xf32> -> !flow.dispatch.tensor<writeonly:tensor<?xf32>>{%arg0_dim0}
      return
    }
  }
}
// CHECK: vm.func private @simple_mul_dynamic
func.func @simple_mul_dynamic(%arg0: tensor<?xf32>, %arg1: tensor<?xf32>) -> tensor<?xf32> {
  %c0 = arith.constant 0 : index
  // CHECK: vm.call @hal.buffer_view.dim
  %arg0_dim0 = tensor.dim %arg0, %c0 : tensor<?xf32>
  // CHECK: vm.call @hal.buffer_view.dim
  %arg1_dim0 = tensor.dim %arg1, %c0 : tensor<?xf32>
  // CHECK: vm.call.variadic @hal.command_buffer.dispatch
  %ret0 = flow.dispatch @executable_2::@dispatch[%arg0_dim0](%arg0, %arg0_dim0, %arg1, %arg1_dim0) : (tensor<?xf32>{%arg0_dim0}, index, tensor<?xf32>{%arg1_dim0}, index) -> tensor<?xf32>{%arg0_dim0}
  return %ret0 : tensor<?xf32>
}
}  // module

// -----

// This shows the same element-wise multiply but without the first level of
// tiling. This will execute in a single workgroup regardless of tensor size
// (though here it's 4 so it wouldn't be distributed anyway).

// CHECK-LABEL: vm.module public @untiled
module @untiled {
// CHECK: vm.rodata private @executable_3_vmvx_bytecode_fb
stream.executable private @executable_3 {
  stream.executable.export public @dispatch workgroups(%arg0: index) -> (index, index, index) {
    %c1 = arith.constant 1 : index
    stream.return %c1, %c1, %c1 : index, index, index
  }
  builtin.module {
    func.func @dispatch(%arg0: !stream.binding, %arg1: !stream.binding, %ret0: !stream.binding) {
      %c0 = arith.constant 0 : index
      %0 = stream.binding.subspan %arg0[%c0] : !stream.binding -> !flow.dispatch.tensor<readonly:tensor<4xf32>>
      %1 = stream.binding.subspan %arg1[%c0] : !stream.binding -> !flow.dispatch.tensor<readonly:tensor<4xf32>>
      %2 = stream.binding.subspan %ret0[%c0] : !stream.binding -> !flow.dispatch.tensor<writeonly:tensor<4xf32>>
      %3 = flow.dispatch.tensor.load %0, offsets = [0], sizes = [4], strides = [1] : !flow.dispatch.tensor<readonly:tensor<4xf32>> -> tensor<4xf32>
      %4 = flow.dispatch.tensor.load %1, offsets = [0], sizes = [4], strides = [1] : !flow.dispatch.tensor<readonly:tensor<4xf32>> -> tensor<4xf32>
      %5 = tensor.empty() : tensor<4xf32>
      %6 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%3, %4 : tensor<4xf32>, tensor<4xf32>) outs(%5 : tensor<4xf32>) {
      ^bb0(%lhs: f32, %rhs: f32, %out: f32):
        %7 = arith.mulf %lhs, %rhs : f32
        linalg.yield %7 : f32
      } -> tensor<4xf32>
      flow.dispatch.tensor.store %6, %2, offsets = [0], sizes = [4], strides = [1] : tensor<4xf32> -> !flow.dispatch.tensor<writeonly:tensor<4xf32>>
      return
    }
  }
}
// CHECK: vm.func private @simple_mul_untiled
func.func @simple_mul_untiled(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) -> tensor<4xf32> {
  %c1 = arith.constant 1 : index
  %ret0 = flow.dispatch @executable_3::@dispatch[%c1](%arg0, %arg1) : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
  return %ret0 : tensor<4xf32>
}
}  // module

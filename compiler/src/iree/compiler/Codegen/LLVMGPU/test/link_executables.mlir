// RUN: iree-opt --pass-pipeline='builtin.module(iree-llvmgpu-link-executables{target="rocm"})' --split-input-file %s | FileCheck %s --check-prefix=CHECK-TARGET
// RUN: iree-opt --pass-pipeline='builtin.module(iree-llvmgpu-link-executables{target="cuda"},iree-llvmgpu-link-executables{target="rocm"})' --split-input-file %s | FileCheck %s --check-prefix=CHECK-MULTI

#executable_target_rocm = #hal.executable.target<"rocm", "rocm-hsaco-fb">

// Expect a single executable with both exports and correct ordinals.
// CHECK-TARGET: hal.executable private @link_executables_linked
// CHECK-TARGET:   hal.executable.variant public @rocm_hsaco_fb
// CHECK-TARGET:     hal.executable.export public @export0 ordinal(0)
// CHECK-TARGET:     hal.executable.export public @export1 ordinal(1)

// Expect one LLVM module with all globals and functions.
// Note that shared memory is duplicated but dynamic shared memory is not.
// CHECK-TARGET: builtin.module
// CHECK-TARGET-NEXT: llvm.mlir.global external @__dynamic_shared_memory__
// CHECK-TARGET-NEXT: llvm.mlir.global private @__shared_memory__{{.+}} : !llvm.array<2 x array<64 x i32>>
// CHECK-TARGET-NEXT: llvm.func @export0
// CHECK-TARGET-NEXT:   llvm.mlir.addressof @__dynamic_shared_memory__ : !llvm.ptr<3>
// CHECK-TARGET-NEXT:   llvm.mlir.addressof @__shared_memory__ : !llvm.ptr<3>
//      CHECK-TARGET: llvm.mlir.global private @__shared_memory___0{{.+}} : !llvm.array<2 x array<128 x i32>>
// CHECK-TARGET-NEXT: llvm.func @export1
// CHECK-TARGET-NEXT:   llvm.mlir.addressof @__dynamic_shared_memory__ : !llvm.ptr<3>
// CHECK-TARGET-NEXT:   llvm.mlir.addressof @__shared_memory___0 : !llvm.ptr<3>

hal.executable private @executable0 {
  hal.executable.variant public @rocm_hsaco_fb target(#executable_target_rocm) {
    hal.executable.export public @export0 ordinal(0) layout(#hal.pipeline.layout<bindings = [#hal.pipeline.binding<storage_buffer>]>) count(%arg0: !hal.device) -> (index, index, index) {
      %c1 = arith.constant 1 : index
      hal.return %c1, %c1, %c1 : index, index, index
    }
    builtin.module {
      llvm.mlir.global external @__dynamic_shared_memory__() {addr_space = 3 : i32, alignment = 16 : i64} : !llvm.array<0 x i8>
      llvm.mlir.global private @__shared_memory__() {addr_space = 3 : i32, alignment = 4 : i64} : !llvm.array<2 x array<64 x i32>>
      llvm.func @export0(%arg0: !llvm.ptr<1> {llvm.align = 16 : i32, llvm.noalias}) {
        %0 = llvm.mlir.addressof @__dynamic_shared_memory__ : !llvm.ptr<3>
        %1 = llvm.mlir.addressof @__shared_memory__ : !llvm.ptr<3>
        llvm.return
      }
    }
  }
}
hal.executable private @executable1 {
  hal.executable.variant public @rocm_hsaco_fb target(#executable_target_rocm) {
    hal.executable.export public @export1 ordinal(0) layout(#hal.pipeline.layout<bindings = [#hal.pipeline.binding<storage_buffer>]>) count(%arg0: !hal.device) -> (index, index, index) {
      %c1 = arith.constant 1 : index
      hal.return %c1, %c1, %c1 : index, index, index
    }
    builtin.module {
      llvm.mlir.global external @__dynamic_shared_memory__() {addr_space = 3 : i32, alignment = 16 : i64} : !llvm.array<0 x i8>
      llvm.mlir.global private @__shared_memory__() {addr_space = 3 : i32, alignment = 4 : i64} : !llvm.array<2 x array<128 x i32>>
      llvm.func @export1(%arg0: !llvm.ptr<1> {llvm.align = 16 : i32, llvm.noalias}) {
        %0 = llvm.mlir.addressof @__dynamic_shared_memory__ : !llvm.ptr<3>
        %1 = llvm.mlir.addressof @__shared_memory__ : !llvm.ptr<3>
        llvm.return
      }
    }
  }
}

// -----

#executable_target_cuda = #hal.executable.target<"cuda", "cuda-nvptx-fb">
#executable_target_rocm = #hal.executable.target<"rocm", "rocm-hsaco-fb">

// Expect only one target be linked when specified.
// CHECK-TARGET: hal.executable private @link_executables_linked
// CHECK-TARGET:   hal.executable.variant public @rocm_hsaco_fb_1
// CHECK-TARGET:     hal.executable.export public @export0 ordinal(0)
// CHECK-TARGET:     hal.executable.export public @export1 ordinal(1)
// CHECK-TARGET: hal.executable private @executable0
// CHECK-TARGET:   hal.executable.variant public @cuda_nvptx_fb
// CHECK-TARGET:     hal.executable.export public @export0 ordinal(0)
// CHECK-TARGET: hal.executable private @executable1
// CHECK-TARGET:   hal.executable.variant public @cuda_nvptx_fb
// CHECK-TARGET:     hal.executable.export public @export1 ordinal(0)

// Multiple applications of the pass per target should not conflict.
// CHECK-MULTI: hal.executable private @link_executables_linked_0
// CHECK-MULTI:   hal.executable.variant public @rocm_hsaco_fb
// CHECK-MULTI:     hal.executable.export public @export0 ordinal(0)
// CHECK-MULTI:     hal.executable.export public @export1 ordinal(1)
// CHECK-MULTI: hal.executable private @link_executables_linked
// CHECK-MULTI:   hal.executable.variant public @cuda_nvptx_fb_0
// CHECK-MULTI:     hal.executable.export public @export0 ordinal(0)
// CHECK-MULTI:     hal.executable.export public @export1 ordinal(1)

hal.executable private @executable0 {
  hal.executable.variant public @cuda_nvptx_fb target(#executable_target_cuda) {
    hal.executable.export public @export0 ordinal(0) layout(#hal.pipeline.layout<bindings = [#hal.pipeline.binding<storage_buffer>]>) count(%arg0: !hal.device) -> (index, index, index) {
      %c1 = arith.constant 1 : index
      hal.return %c1, %c1, %c1 : index, index, index
    }
    builtin.module {
      llvm.func @export0(%arg0: !llvm.ptr<1> {llvm.align = 16 : i32, llvm.noalias}) {
        llvm.return
      }
    }
  }
  hal.executable.variant public @rocm_hsaco_fb target(#executable_target_rocm) {
    hal.executable.export public @export0 ordinal(0) layout(#hal.pipeline.layout<bindings = [#hal.pipeline.binding<storage_buffer>]>) count(%arg0: !hal.device) -> (index, index, index) {
      %c1 = arith.constant 1 : index
      hal.return %c1, %c1, %c1 : index, index, index
    }
    builtin.module {
      llvm.func @export0(%arg0: !llvm.ptr<1> {llvm.align = 16 : i32, llvm.noalias}) {
        llvm.return
      }
    }
  }
}
hal.executable private @executable1 {
  hal.executable.variant public @cuda_nvptx_fb target(#executable_target_cuda) {
    hal.executable.export public @export1 ordinal(0) layout(#hal.pipeline.layout<bindings = [#hal.pipeline.binding<storage_buffer>]>) count(%arg0: !hal.device) -> (index, index, index) {
      %c1 = arith.constant 1 : index
      hal.return %c1, %c1, %c1 : index, index, index
    }
    builtin.module {
      llvm.func @export1(%arg0: !llvm.ptr<1> {llvm.align = 16 : i32, llvm.noalias}) {
        llvm.return
      }
    }
  }
  hal.executable.variant public @rocm_hsaco_fb target(#executable_target_rocm) {
    hal.executable.export public @export1 ordinal(0) layout(#hal.pipeline.layout<bindings = [#hal.pipeline.binding<storage_buffer>]>) count(%arg0: !hal.device) -> (index, index, index) {
      %c1 = arith.constant 1 : index
      hal.return %c1, %c1, %c1 : index, index, index
    }
    builtin.module {
      llvm.func @export1(%arg0: !llvm.ptr<1> {llvm.align = 16 : i32, llvm.noalias}) {
        llvm.return
      }
    }
  }
}

// -----

// Tests that externally defined executables (no inner module) don't get linked
// into internal ones (inner module).

hal.executable private @internal_executable {
  hal.executable.variant public @rocm_hsaco_fb target(<"rocm", "rocm-hsaco-fb">) {
    hal.executable.export public @export0 ordinal(0) layout(#hal.pipeline.layout<bindings = [#hal.pipeline.binding<storage_buffer>]>) count(%arg0: !hal.device) -> (index, index, index) {
      %c1 = arith.constant 1 : index
      hal.return %c1, %c1, %c1 : index, index, index
    }
  }
}
hal.executable private @external_executable {
  hal.executable.variant public @rocm_hsaco_fb_0 target(<"rocm", "rocm-hsaco-fb">) {
    hal.executable.export public @export1 ordinal(0) layout(#hal.pipeline.layout<bindings = [#hal.pipeline.binding<storage_buffer>]>) count(%arg0: !hal.device) -> (index, index, index) {
      %c1 = arith.constant 1 : index
      hal.return %c1, %c1, %c1 : index, index, index
    }
    builtin.module {
    }
  }
}

// CHECK-LABEL: hal.executable private
//       CHECK:   hal.executable.export public @export1
//   CHECK-NOT:   hal.executable.export public @export0
//       CHECK:   builtin.module
// CHECK-LABEL: hal.executable private @external_executable

// -----

// Tests that externally defined executables (no inner module) don't get linked
// into internal ones (inner module).

hal.executable private @internal_executable {
  hal.executable.variant public @rocm_hsaco_fb target(<"rocm", "rocm-hsaco-fb">) {
    hal.executable.export public @export0 ordinal(0) layout(#hal.pipeline.layout<bindings = [#hal.pipeline.binding<storage_buffer>]>) count(%arg0: !hal.device) -> (index, index, index) {
      %c1 = arith.constant 1 : index
      hal.return %c1, %c1, %c1 : index, index, index
    }
    builtin.module {
    }
  }
}
hal.executable private @external_executable {
  hal.executable.variant public @rocm_hsaco_fb_0 target(<"rocm", "rocm-hsaco-fb">) {
    hal.executable.export public @export1 ordinal(0) layout(#hal.pipeline.layout<bindings = [#hal.pipeline.binding<storage_buffer>]>) count(%arg0: !hal.device) -> (index, index, index) {
      %c1 = arith.constant 1 : index
      hal.return %c1, %c1, %c1 : index, index, index
    }
  }
}
// CHECK-LABEL: hal.executable private
//       CHECK:   hal.executable.export public @export0
//   CHECK-NOT:   hal.executable.export public @export1
//       CHECK:   builtin.module
// CHECK-LABEL: hal.executable private @external_executable

// -----

// Tests that any variant being externally defined disables linking.
hal.executable private @internal_executable {
  hal.executable.variant public @rocm_hsaco_fb target(<"rocm", "rocm-hsaco-fb">) {
    hal.executable.export public @export0 ordinal(0) layout(#hal.pipeline.layout<bindings = [#hal.pipeline.binding<storage_buffer>]>) count(%arg0: !hal.device) -> (index, index, index) {
      %c1 = arith.constant 1 : index
      hal.return %c1, %c1, %c1 : index, index, index
    }
    builtin.module {
    }
  }
}
hal.executable private @external_executable {
  hal.executable.variant public @rocm_hsaco_fb_0 target(<"rocm", "rocm-hsaco-fb">) {
    hal.executable.export public @export1 ordinal(0) layout(#hal.pipeline.layout<bindings = [#hal.pipeline.binding<storage_buffer>]>) count(%arg0: !hal.device) -> (index, index, index) {
      %c1 = arith.constant 1 : index
      hal.return %c1, %c1, %c1 : index, index, index
    }
  }
  hal.executable.variant public @rocm_hsaco_fb_1 target(<"rocm", "rocm-hsaco-fb">) {
    hal.executable.export public @export1 ordinal(0) layout(#hal.pipeline.layout<bindings = [#hal.pipeline.binding<storage_buffer>]>) count(%arg0: !hal.device) -> (index, index, index) {
      %c1 = arith.constant 1 : index
      hal.return %c1, %c1, %c1 : index, index, index
    }
    builtin.module {
    }
  }
}

// CHECK-LABEL: hal.executable private
//       CHECK:   hal.executable.export public @export0
//   CHECK-NOT:   hal.executable.export public @export1
//       CHECK:   builtin.module
// CHECK-LABEL: hal.executable private @external_executable

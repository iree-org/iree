// RUN: iree-opt --split-input-file --pass-pipeline="builtin.module(hal.executable(hal.executable.variant(iree-codegen-propagate-dispatch-config)))" --verify-diagnostics %s | FileCheck %s

// Basic: single dispatch_config + export.
hal.executable private @basic_exe {
  hal.executable.variant public @rocm target(<"rocm", "rocm-hsaco-fb">) {
    hal.executable.export @matmul ordinal(0)
        layout(#hal.pipeline.layout<bindings = [
          #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">,
          #hal.pipeline.binding<storage_buffer, Indirect>
        ]>)
        count(%device: !hal.device, %w0: index, %w1: index)
            -> (index, index, index) {
      %x, %y, %z = iree_tensor_ext.dispatch.workgroup_count_from_slice(%w0, %w1)
      hal.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @matmul() {
        return
      }
      iree_codegen.dispatch_config @matmul
          workgroup_size = [64, 16, 1] subgroup_size = 64 {
        ^bb0(%w0: index, %w1: index):
          %c1 = arith.constant 1 : index
          %0 = affine.apply affine_map<()[s0] -> (s0 ceildiv 128)>()[%w0]
          iree_codegen.yield %0, %w1, %c1 : index, index, index
      }
    }
  }
}
// CHECK-LABEL: hal.executable private @basic_exe
//       CHECK:   hal.executable.export public @matmul
//       CHECK:     count(%[[DEV:.+]]: !hal.device, %[[W0:.+]]: index, %[[W1:.+]]: index)
//       CHECK:       %[[C1:.+]] = arith.constant 1 : index
//       CHECK:       %[[X:.+]] = affine.apply
//       CHECK:       hal.return %[[X]], %[[W1]], %[[C1]]
//       CHECK:     } attributes {subgroup_size = 64 : index, workgroup_size = [64 : index, 16 : index, 1 : index]}
//       CHECK:   builtin.module
//       CHECK:     func.func @matmul()
//   CHECK-NOT:     iree_codegen.dispatch_config

// -----

// Multiple dispatch_configs (specialization).
hal.executable private @specialized_exe {
  hal.executable.variant public @rocm target(<"rocm", "rocm-hsaco-fb">) {
    hal.executable.export @matmul ordinal(0)
        layout(#hal.pipeline.layout<bindings = [
          #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">,
          #hal.pipeline.binding<storage_buffer, Indirect>
        ]>)
        count(%device: !hal.device, %w0: index)
            -> (index, index, index) {
      %x, %y, %z = iree_tensor_ext.dispatch.workgroup_count_from_slice(%w0)
      hal.return %x, %y, %z : index, index, index
    }
    hal.executable.export @matmul_0 ordinal(1)
        layout(#hal.pipeline.layout<bindings = [
          #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">,
          #hal.pipeline.binding<storage_buffer, Indirect>
        ]>)
        count(%device: !hal.device, %w0: index)
            -> (index, index, index) {
      %x, %y, %z = iree_tensor_ext.dispatch.workgroup_count_from_slice(%w0)
      hal.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @matmul() {
        return
      }
      func.func @matmul_0() {
        return
      }
      iree_codegen.dispatch_config @matmul
          workgroup_size = [64, 16, 1] subgroup_size = 64 {
        ^bb0(%w0: index):
          %c1 = arith.constant 1 : index
          iree_codegen.yield %w0, %c1, %c1 : index, index, index
      }
      iree_codegen.dispatch_config @matmul_0
          workgroup_size = [256, 1, 1] subgroup_size = 64 {
        ^bb0(%w0: index):
          %c1 = arith.constant 1 : index
          %0 = affine.apply affine_map<()[s0] -> (s0 ceildiv 64)>()[%w0]
          iree_codegen.yield %0, %c1, %c1 : index, index, index
      }
    }
  }
}
// CHECK-LABEL: hal.executable private @specialized_exe
//       CHECK:   hal.executable.export public @matmul
//       CHECK:     count(%{{.+}}: !hal.device, %[[W0A:.+]]: index)
//       CHECK:       %[[C1A:.+]] = arith.constant 1 : index
//       CHECK:       hal.return %[[W0A]], %[[C1A]], %[[C1A]]
//       CHECK:     } attributes {subgroup_size = 64 : index, workgroup_size = [64 : index, 16 : index, 1 : index]}
//       CHECK:   hal.executable.export public @matmul_0
//       CHECK:     count(%{{.+}}: !hal.device, %[[W0B:.+]]: index)
//       CHECK:       %[[C1B:.+]] = arith.constant 1 : index
//       CHECK:       %[[X:.+]] = affine.apply
//       CHECK:       hal.return %[[X]], %[[C1B]], %[[C1B]]
//       CHECK:     } attributes {subgroup_size = 64 : index, workgroup_size = [256 : index, 1 : index, 1 : index]}

// -----

// No subgroup_size attribute.
hal.executable private @no_subgroup_exe {
  hal.executable.variant public @rocm target(<"rocm", "rocm-hsaco-fb">) {
    hal.executable.export @entry ordinal(0)
        layout(#hal.pipeline.layout<bindings = [
          #hal.pipeline.binding<storage_buffer, Indirect>
        ]>)
        count(%device: !hal.device, %w0: index)
            -> (index, index, index) {
      %x, %y, %z = iree_tensor_ext.dispatch.workgroup_count_from_slice(%w0)
      hal.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @entry() {
        return
      }
      iree_codegen.dispatch_config @entry
          workgroup_size = [1024, 1, 1] {
        ^bb0(%w0: index):
          %c1 = arith.constant 1 : index
          iree_codegen.yield %w0, %c1, %c1 : index, index, index
      }
    }
  }
}
// CHECK-LABEL: hal.executable private @no_subgroup_exe
//       CHECK:   hal.executable.export public @entry
//       CHECK:       hal.return
//       CHECK:     } attributes {workgroup_size = [1024 : index, 1 : index, 1 : index]}
//   CHECK-NOT:     subgroup_size

// -----

// No dispatch_config ops — pass is a no-op.
hal.executable private @noop_exe {
  hal.executable.variant public @rocm target(<"rocm", "rocm-hsaco-fb">) {
    hal.executable.export @entry ordinal(0)
        layout(#hal.pipeline.layout<bindings = [
          #hal.pipeline.binding<storage_buffer, Indirect>
        ]>)
        count(%device: !hal.device, %w0: index)
            -> (index, index, index) {
      %x, %y, %z = iree_tensor_ext.dispatch.workgroup_count_from_slice(%w0)
      hal.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @entry() {
        return
      }
    }
  }
}
// CHECK-LABEL: hal.executable private @noop_exe
//       CHECK:   iree_tensor_ext.dispatch.workgroup_count_from_slice

// -----

// Error: arity mismatch.
hal.executable private @arity_mismatch_exe {
  hal.executable.variant public @rocm target(<"rocm", "rocm-hsaco-fb">) {
    hal.executable.export @entry ordinal(0)
        layout(#hal.pipeline.layout<bindings = [
          #hal.pipeline.binding<storage_buffer, Indirect>
        ]>)
        count(%device: !hal.device, %w0: index)
            -> (index, index, index) {
      %x, %y, %z = iree_tensor_ext.dispatch.workgroup_count_from_slice(%w0)
      hal.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @entry() {
        return
      }
      // expected-error @+1 {{workload arity mismatch}}
      iree_codegen.dispatch_config @entry
          workgroup_size = [64, 1, 1] {
        ^bb0(%w0: index, %w1: index, %w2: index):
          %c1 = arith.constant 1 : index
          iree_codegen.yield %w0, %c1, %c1 : index, index, index
      }
    }
  }
}

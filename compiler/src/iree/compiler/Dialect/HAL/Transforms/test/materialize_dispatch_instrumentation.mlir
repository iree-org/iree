// RUN: iree-opt --split-input-file --pass-pipeline='builtin.module(iree-hal-materialize-dispatch-instrumentation{bufferSize=64mib})' %s | FileCheck %s

module attributes {hal.device.targets = [
  #hal.device.target<"llvm-cpu", {
    executable_targets = [
      #hal.executable.target<"llvm-cpu", "embedded-elf-arm_64">,
      #hal.executable.target<"llvm-cpu", "embedded-elf-x86_64">
    ]
  }>
]} {

  // Instrumentation storage buffer allocated at startup (defaults to 64MB + footer):
  // CHECK: util.global public @__dispatch_instrumentation : !stream.resource<external>
  // CHECK: util.initializer
  // CHECK:   %[[DEFAULT_SIZE:.+]] = arith.constant 67112960
  // CHECK:   %[[ALLOC_BUFFER:.+]] = stream.resource.alloc uninitialized : !stream.resource<external>{%[[DEFAULT_SIZE]]}
  // CHECK:   util.global.store %[[ALLOC_BUFFER]], @__dispatch_instrumentation

  // Query function used by tools to get the buffers and metadata:
  // CHECK: func.func @__query_instruments(%[[LIST:.+]]: !util.list<?>)
  // CHECK:   %[[INTERNAL_BUFFER:.+]] = util.global.load @__dispatch_instrumentation
  // CHECK:   %[[EXPORTED_BUFFER:.+]] = stream.tensor.export %[[INTERNAL_BUFFER]]
  // CHECK:   util.list.set %[[LIST]]{{.+}}
  // CHECK:   util.list.set %[[LIST]]{{.+}}
  // CHECK:   util.list.set %[[LIST]]{{.+}}
  // CHECK:   util.list.set %[[LIST]]{{.+}}
  // CHECK:   util.list.set %[[LIST]]{{.+}}, %[[EXPORTED_BUFFER]]

  stream.executable private @executable {
    stream.executable.export public @dispatch workgroups() -> (index, index, index) {
      %x, %y, %z = flow.dispatch.workgroup_count_from_dag_root
      stream.return %x, %y, %z : index, index, index
    }
    builtin.module {
      // Dispatches get the instrumentation buffer and a unique dispatch site ID:
      // CHECK: func.func @dispatch
      // CHECK-SAME: (%arg0: !stream.binding {stream.alignment = 64 : index}, %arg1: !stream.binding {stream.alignment = 64 : index}, %[[INSTR_BINDING:.+]]: !stream.binding {stream.alignment = 64 : index}, %[[SITE_ID:.+]]: i32)
      func.func @dispatch(%arg0: !stream.binding {stream.alignment = 64 : index}, %arg1: !stream.binding {stream.alignment = 64 : index}) {
        // Default instrumentation just adds the workgroup marker.
        // Subsequent dispatch instruments will use the workgroup key.
        // CHECK: %[[INSTR_BUFFER:.+]] = stream.binding.subspan %[[INSTR_BINDING]]
        // CHECK: %[[WORKGROUP_KEY:.+]] = hal.instrument.workgroup[%[[INSTR_BUFFER]] : memref<67112960xi8>] dispatch(%[[SITE_ID]]) : index
        %c0 = arith.constant 0 : index
        %cst = arith.constant 2.000000e+00 : f32
        %0 = stream.binding.subspan %arg0[%c0] : !stream.binding -> !flow.dispatch.tensor<readonly:tensor<f32>>
        %1 = stream.binding.subspan %arg1[%c0] : !stream.binding -> !flow.dispatch.tensor<writeonly:tensor<f32>>
        %2 = flow.dispatch.tensor.load %0, offsets = [], sizes = [], strides = [] : !flow.dispatch.tensor<readonly:tensor<f32>> -> tensor<f32>
        %3 = tensor.empty() : tensor<f32>
        %4 = linalg.generic {indexing_maps = [affine_map<() -> ()>, affine_map<() -> ()>], iterator_types = []} ins(%2 : tensor<f32>) outs(%3 : tensor<f32>) {
        ^bb0(%in: f32, %out: f32):
          %5 = math.powf %in, %cst : f32
          linalg.yield %5 : f32
        } -> tensor<f32>
        flow.dispatch.tensor.store %4, %1, offsets = [], sizes = [], strides = [] : tensor<f32> -> !flow.dispatch.tensor<writeonly:tensor<f32>>
        return
      }
    }
  }
  func.func @main(%arg0: !stream.resource<external>) -> !stream.resource<external> {
    %c0 = arith.constant 0 : index
    %c128 = arith.constant 128 : index
    %ret0 = stream.resource.alloc uninitialized : !stream.resource<external>{%c128}
    // The instrumentation buffer is captured by submissions for dispatch.
    // Note that there's no synchronization here (no timepoint waits/etc) as
    // all accesses to the buffer are atomic.
    // CHECK: %[[EXECUTE_BUFFER:.+]] = util.global.load @__dispatch_instrumentation
    // CHECK: stream.cmd.execute
    // CHECK-SAME: %[[EXECUTE_BUFFER]] as %[[CAPTURE_BUFFER:.+]]: !stream.resource<external>{%[[DEFAULT_SIZE]]})
    %timepoint = stream.cmd.execute with(%arg0 as %arg0_capture: !stream.resource<external>{%c128}, %ret0 as %ret0_capture: !stream.resource<external>{%c128}) {
      // CHECK: stream.cmd.dispatch @executable::@dispatch
      stream.cmd.dispatch @executable::@dispatch {
        ro %arg0_capture[%c0 for %c128] : !stream.resource<external>{%c128},
        wo %ret0_capture[%c0 for %c128] : !stream.resource<external>{%c128}
        // CHECK: rw %[[CAPTURE_BUFFER]]
      }
    } => !stream.timepoint
    %ret0_ready = stream.timepoint.await %timepoint => %ret0 : !stream.resource<external>{%c128}
    return %ret0_ready : !stream.resource<external>
  }
}

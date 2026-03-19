// RUN: iree-opt --split-input-file --pass-pipeline="builtin.module(hal.executable(hal.executable.variant(iree-codegen-create-dispatch-config)))" \
// RUN:   %s | FileCheck %s

// Export with from_slice count region only, which is a basic test.
#pipeline_layout = #hal.pipeline.layout<constants = 2, bindings = [
  #hal.pipeline.binding<storage_buffer>
]>
hal.executable private @from_slice {
  hal.executable.variant public @variant target(#hal.executable.target<"", "">) {
    hal.executable.export public @entry_point layout(#pipeline_layout)
        count(%device: !hal.device, %arg0: index, %arg1: index) -> (index, index, index) {
      %x, %y, %z = iree_tensor_ext.dispatch.workgroup_count_from_slice(%arg0, %arg1)
      hal.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @entry_point() {
        return
      }
    }
  }
}
// CHECK-LABEL: hal.executable private @from_slice
//       CHECK:   func.func @entry_point
//       CHECK:   iree_codegen.dispatch_config @entry_point
//   CHECK-NOT:     workgroup_size
//       CHECK:     ^bb0(%[[ARG0:.+]]: index, %[[ARG1:.+]]: index):
//       CHECK:       %[[X:.+]], %[[Y:.+]], %[[Z:.+]] = iree_tensor_ext.dispatch.workgroup_count_from_slice(%[[ARG0]], %[[ARG1]])
//       CHECK:       iree_codegen.yield %[[X]], %[[Y]], %[[Z]]

// -----

// Export with from_slice + split_reduction_modifier.
#pipeline_layout = #hal.pipeline.layout<constants = 6, bindings = [
  #hal.pipeline.binding<storage_buffer, "ReadOnly">,
  #hal.pipeline.binding<storage_buffer>
]>
hal.executable private @split_reduction {
  hal.executable.variant public @variant target(#hal.executable.target<"", "">) {
    hal.executable.export public @entry_point layout(#pipeline_layout)
        count(%device: !hal.device, %arg0: index, %arg1: index, %arg2: index,
              %arg3: index, %arg4: index, %arg5: index) -> (index, index, index) {
      %x, %y, %z = iree_tensor_ext.dispatch.workgroup_count_from_slice(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5)
      %rx, %ry, %rz = iree_tensor_ext.dispatch.workgroup_count_split_reduction_modifier
          workgroups(%x, %y, %z) workload(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5)
      hal.return %rx, %ry, %rz : index, index, index
    }
    builtin.module {
      func.func @entry_point() {
        return
      }
    }
  }
}
// CHECK-LABEL: hal.executable private @split_reduction
//       CHECK:   func.func @entry_point
//       CHECK:   iree_codegen.dispatch_config @entry_point
//   CHECK-NOT:     workgroup_size
//       CHECK:     ^bb0(%[[W0:.+]]: index, %[[W1:.+]]: index, %[[W2:.+]]: index, %[[W3:.+]]: index, %[[W4:.+]]: index, %[[W5:.+]]: index):
//       CHECK:       %[[X:.+]], %[[Y:.+]], %[[Z:.+]] = iree_tensor_ext.dispatch.workgroup_count_from_slice(%[[W0]], %[[W1]], %[[W2]], %[[W3]], %[[W4]], %[[W5]])
//       CHECK:       iree_tensor_ext.dispatch.workgroup_count_split_reduction_modifier workgroups(%[[X]], %[[Y]], %[[Z]]) workload(%[[W0]], %[[W1]], %[[W2]], %[[W3]], %[[W4]], %[[W5]])
//       CHECK:       iree_codegen.yield

// -----

// Export with no count region — dispatch_config with stub body, no workgroup_size.
#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>
]>
hal.executable private @no_count_region {
  hal.executable.variant public @variant target(#hal.executable.target<"", "">) {
    hal.executable.export public @entry_point layout(#pipeline_layout)
    builtin.module {
      func.func @entry_point() {
        return
      }
    }
  }
}
// CHECK-LABEL: hal.executable private @no_count_region
//       CHECK:   func.func @entry_point
//       CHECK:   iree_codegen.dispatch_config @entry_point
//   CHECK-NOT:     workgroup_size
//       CHECK:     %[[C1:.+]] = arith.constant 1 : index
//       CHECK:     iree_codegen.yield %[[C1]], %[[C1]], %[[C1]]

// -----

// Test that dispatch_config placed after its function.
#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>
]>
hal.executable private @interleaving {
  hal.executable.variant public @variant target(#hal.executable.target<"", "">) {
    hal.executable.export public @fn1 layout(#pipeline_layout)
        count(%device: !hal.device, %arg0: index) -> (index, index, index) {
      %x, %y, %z = iree_tensor_ext.dispatch.workgroup_count_from_slice(%arg0)
      hal.return %x, %y, %z : index, index, index
    }
    hal.executable.export public @fn2 layout(#pipeline_layout)
        count(%device: !hal.device, %arg0: index, %arg1: index) -> (index, index, index) {
      %x, %y, %z = iree_tensor_ext.dispatch.workgroup_count_from_slice(%arg0, %arg1)
      hal.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @fn1() {
        return
      }
      func.func @fn2() {
        return
      }
    }
  }
}
// CHECK-LABEL: hal.executable private @interleaving
//       CHECK:   func.func @fn1
//       CHECK:   iree_codegen.dispatch_config @fn1
//       CHECK:   func.func @fn2
//       CHECK:   iree_codegen.dispatch_config @fn2

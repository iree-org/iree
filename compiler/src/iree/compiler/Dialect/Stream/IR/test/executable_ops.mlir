// RUN: iree-opt --split-input-file %s --verify-diagnostics | FileCheck %s

// CHECK-LABEL: stream.executable private @executable
stream.executable private @executable {
  // CHECK-NEXT: stream.executable.export public @dispatch
  stream.executable.export public @dispatch
  // CHECK-NEXT: builtin.module
  builtin.module {
    // CHECK-NEXT: util.func private @dispatch(%arg0: !stream.binding, %arg1: !stream.binding, %arg2: index) {
    util.func private @dispatch(%arg0: !stream.binding, %arg1: !stream.binding, %arg2: index) {
      %c0 = arith.constant 0 : index
      // CHECK-DAG: = stream.binding.subspan %arg0[%c0] : !stream.binding -> !flow.dispatch.tensor<readwrite:tensor<?x5x64xf32>>{%arg2}
      %0 = stream.binding.subspan %arg0[%c0] : !stream.binding -> !flow.dispatch.tensor<readwrite:tensor<?x5x64xf32>>{%arg2}
      // CHECK-DAG: = stream.binding.subspan %arg1[%c0] : !stream.binding -> !flow.dispatch.tensor<readonly:tensor<?x5x4xf32>>{%arg2}
      %1 = stream.binding.subspan %arg1[%c0] : !stream.binding -> !flow.dispatch.tensor<readonly:tensor<?x5x4xf32>>{%arg2}
      // CHECK: util.return
      util.return
    }
  }
}

// -----

// CHECK-LABEL: stream.executable private @executable_with_workgroup_count
stream.executable private @executable_with_workgroup_count {
  // CHECK-NEXT: stream.executable.export public @dispatch
  stream.executable.export public @dispatch
      // CHECK-SAME: workgroups(%arg0: index, %arg1: index, %arg2: index) -> (index, index, index) {
      workgroups(%arg0: index, %arg1: index, %arg2: index) -> (index, index, index) {
        // CHECK-NEXT: stream.return %arg0, %arg1, %arg2 : index, index, index
        stream.return %arg0, %arg1, %arg2 : index, index, index
      }
  // CHECK: builtin.module
  builtin.module {
    // CHECK-NEXT: util.func private @dispatch
    util.func private @dispatch() {
      // CHECK: util.return
      util.return
    }
  }
}

// -----

stream.executable private @bad_workgroup_result_count {
  // expected-error @+1 {{workgroup count region must return the XYZ dimension counts}}
  stream.executable.export public @dispatch workgroups(%arg0: index, %arg1: index) -> (index, index) {
    stream.return %arg0, %arg1 : index, index
  }
  builtin.module  {
    util.func private @dispatch() {
      util.return
    }
  }
}

// -----

stream.executable private @bad_workgroup_result_types {
  // expected-error @+1 {{workgroup count region must return the XYZ dimension counts}}
  stream.executable.export public @dispatch workgroups(%arg0: index, %arg1: f32) -> (index, f32, index) {
    stream.return %arg0, %arg1, %arg0 : index, f32, index
  }
  builtin.module  {
    util.func private @dispatch() {
      util.return
    }
  }
}

// -----

stream.executable private @executable {
  stream.executable.export public @dispatch
  builtin.module {
    util.func private @dispatch(%arg0: !stream.binding, %arg1: index) {
      %c0 = arith.constant 0 : index
      %0 = stream.binding.subspan %arg0[%c0] : !stream.binding -> !flow.dispatch.tensor<readwrite:tensor<?x5x64xf32>>{%arg1}
      util.return
    }
  }
}

util.func private @cmdDispatchExecutableSignatureMismatch(%arg0: !stream.resource<transient>,
                                                  %arg1: index,
                                                  %arg2: !stream.resource<external>,
                                                  %arg3: index) -> !stream.timepoint {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c128 = arith.constant 128 : index
  %0 = stream.cmd.execute with(%arg0 as %arg4: !stream.resource<transient>{%arg1}, %arg2 as %arg5: !stream.resource<external>{%arg3}) {
    // expected-error @+1 {{function type mismatch; expected 2 binding arguments on exported function, but has 1}}
    stream.cmd.dispatch {@executable::@dispatch}[%c1](%c2 : index) {
      ro %arg4[%c0 for %c128] : !stream.resource<transient>{%arg1},
      wo %arg5[%c0 for %c128] : !stream.resource<external>{%arg3}
    }
  } => !stream.timepoint
  util.return %0 : !stream.timepoint
}

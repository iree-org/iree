// RUN: iree-opt --split-input-file %s --verify-diagnostics | FileCheck %s

stream.executable private @executable {
  stream.executable.export public @dispatch
  builtin.module {
    func.func @dispatch(%arg0: !stream.binding, %arg1: index) {
      %c0 = arith.constant 0 : index
      %0 = stream.binding.subspan %arg0[%c0] : !stream.binding -> !flow.dispatch.tensor<readwrite:tensor<?x5x64xf32>>{%arg1}
      return
    }
  }
}

func.func @cmdDispatch(%arg0: !stream.resource<transient>, %arg1: index, %arg2: !stream.resource<external>, %arg3: index) -> !stream.timepoint {
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
  return %0 : !stream.timepoint
}

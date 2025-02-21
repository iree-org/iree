// RUN: iree-compile --compile-to=global-optimization %s \
// RUN:   | FileCheck %s --check-prefix=NO-STRIP-CHECK
// RUN: iree-compile --compile-to=global-optimization \
// RUN:              --iree-global-optimization-opt-level=O2 \
// RUN:              --iree-opt-strip-assertions=false %s \
// RUN:   | FileCheck %s --check-prefix=NO-STRIP-CHECK
// RUN: iree-compile --compile-to=global-optimization \
// RUN:              --iree-opt-level=O2 \
// RUN:              --iree-opt-strip-assertions=false %s \
// RUN:   | FileCheck %s --check-prefix=NO-STRIP-CHECK
// RUN: iree-compile --compile-to=global-optimization \
// RUN:              --iree-opt-level=O2 \
// RUN:              --iree-global-optimization-opt-level=O0 %s \
// RUN:   | FileCheck %s --check-prefix=NO-STRIP-CHECK
//
// RUN: iree-compile --compile-to=global-optimization \
// RUN:              --iree-opt-strip-assertions %s \
// RUN:   | FileCheck %s --check-prefix=STRIP-CHECK
// RUN: iree-compile --compile-to=global-optimization \
// RUN:              --iree-opt-level=O2 %s \
// RUN:   | FileCheck %s --check-prefix=STRIP-CHECK
// RUN: iree-compile --compile-to=global-optimization \
// RUN:              --iree-global-optimization-opt-level=O2 %s \
// RUN:   | FileCheck %s --check-prefix=STRIP-CHECK

util.func public @main(%0 : i1){
  cf.assert %0, "assert"
  util.return
}

//    NO-STRIP-CHECK: util.func
//    NO-STRIP-CHECK:   cf.assert
//       STRIP-CHECK: util.func
//   STRIP-CHECK-NOT:   cf.assert

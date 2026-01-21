// RUN: iree-opt %s --pass-pipeline="builtin.module(func.func(iree-pcf-resolve-tokens, iree-pcf-convert-sref-to-memref, iree-pcf-lower-structural-pcf))" --split-input-file | FileCheck %s

func.func @pcf_workgroup_loop(%arg0: memref<64xf32>) {
  %c64 = arith.constant 64 : index
  pcf.loop scope(#iree_codegen.workgroup_scope) count(%c64)
    execute[%iv: index] {
    %c0 = arith.constant 0.0 : f32
    memref.store %c0, %arg0[%iv] : memref<64xf32>
    pcf.return
  }
  return
}

// CHECK-LABEL: @pcf_workgroup_loop
//       CHECK:   hal.interface.workgroup.id
//       CHECK:   hal.interface.workgroup.count
//       CHECK:   scf.for

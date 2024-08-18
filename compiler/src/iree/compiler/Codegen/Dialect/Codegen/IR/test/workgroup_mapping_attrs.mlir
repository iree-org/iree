// RUN: iree-opt --split-input-file --verify-diagnostics --allow-unregistered-dialect %s | FileCheck %s

func.func @roundtrip() {
  "dummy.op"() {
    workgroup_mapping = [
      #iree_codegen.workgroup_mapping<x>,
      #iree_codegen.workgroup_mapping<y>,
      #iree_codegen.workgroup_mapping<z:0>,
      #iree_codegen.workgroup_mapping<z:1>]
  } : () -> ()
  return
}
// CHECK-LABEL: func @roundtrip()
//       CHECK:   #iree_codegen.workgroup_mapping<x>
//  CHECK-SAME:   #iree_codegen.workgroup_mapping<y>
//  CHECK-SAME:   #iree_codegen.workgroup_mapping<z>
//  CHECK-SAME:   #iree_codegen.workgroup_mapping<z:1>

// -----

func.func @illegal_x_linearized_dim() {
  "dummy.op"() {
    // expected-error @+1 {{illegal to use `delinearizationDim` for x}}
    workgroup_mapping = #iree_codegen.workgroup_mapping<x:1>
  } : () -> ()
  return
}

// -----

func.func @illegal_y_linearized_dim() {
  "dummy.op"() {
    // expected-error @+1 {{illegal to use `delinearizationDim` for y}}
    workgroup_mapping = #iree_codegen.workgroup_mapping<y:1>
  } : () -> ()
  return
}

// RUN: iree-opt -split-input-file -pass-pipeline='iree-convert-std-to-vm' %s | IreeFileCheck %s

// -----
// CHECK-LABEL: @t001_iree_reflection
module @t001_iree_reflection {

module {
  // CHECK: func @t001_iree_reflection
  // CHECK-SAME: iree.reflection = {f = "FOOBAR"}
  func @t001_iree_reflection(%arg0: i32) -> (i32) attributes {iree.reflection = {f = "FOOBAR"}}
  {
    return %arg0 : i32
  }
}

}

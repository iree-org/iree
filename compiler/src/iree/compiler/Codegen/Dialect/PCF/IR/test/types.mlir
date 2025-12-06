// RUN: iree-opt --split-input-file %s | iree-opt --split-input-file | FileCheck %s

util.func private @shaped_ref_with_no_sync(!pcf.sref<1x?x3x?xi32, #pcf.dummy_scope>)
// CHECK: @shaped_ref_with_no_sync
// CHECK-SAME: !pcf.sref<1x?x3x?xi32, #pcf.dummy_scope>

util.func private @shaped_ref_with_type_sync(!pcf.sref<1x?x3x?xi32, #pcf.dummy_scope, i32>)
// CHECK: @shaped_ref_with_type_sync
// CHECK-SAME: !pcf.sref<1x?x3x?xi32, #pcf.dummy_scope, i32>

util.func private @shaped_ref_with_attr_sync(!pcf.sref<1x?x3x?xi32, #pcf.dummy_scope, 42>)
// CHECK: @shaped_ref_with_attr_sync
// CHECK-SAME: !pcf.sref<1x?x3x?xi32, #pcf.dummy_scope, 42 : i64>

util.func private @shaped_ref_with_parent_sync(!pcf.sref<1x?x3x?xi32, sync(#pcf.dummy_scope)>)
// CHECK: @shaped_ref_with_parent_sync
// CHECK-SAME: !pcf.sref<1x?x3x?xi32, sync(#pcf.dummy_scope)>

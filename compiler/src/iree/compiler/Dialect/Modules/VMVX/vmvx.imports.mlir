// VMVX (Virtual Machine-based Vector eXtensions) runtime module imports.
//
// This is embedded in the compiler binary and inserted into any module
// containing VMVX dialect ops (vmvx.*) that is lowered to the VM dialect.
//
// Element types are embedded in the function. The convention used (mostly)
// follows MLIR type names:
// * 'x' : don't-care, bit-depth only.
// * 'i' : signless integer (+ bit depth)   ex: i1 i8 i16 i32 i64
// * 'si': signed integer (+ bit depth)     ex: si32 ...
// * 'ui': unsigned integer (+ bit depth)   ex: ui32 ...
// * 'f' : IREE float (+ bit depth)         ex: f32 f64
//
// See the README.md for more more details on the implementation.
//
// NOTE: each method added here requires a corresponding method in
// `iree/modules/vmvx/exports.inl` and `iree/modules/vmvx/module.c`.
//
// NOTE: there's a maintenance burden to adding new ops as they may have to be
// carried around forever. Always try to convert to the ops that exist unless
// it's performance critical - a few lines of a conversion pattern saves future
// us a lot of pain and breaking changes.
//
// NOTE: experimental functions that are not yet ready to be parts of the core
// module must be prefixed with `ex.` like `vmvx.ex.my_test_op`.
vm.module @vmvx {

//===----------------------------------------------------------------------===//
// VMVX Ops: TODO
//===----------------------------------------------------------------------===//

}  // module

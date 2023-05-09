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
// VMVX Binary Elementwise Kernels
// Each is specialized by opcode, rank and type width.
//===----------------------------------------------------------------------===//

vm.import private @add.2d.f32(
  %lhs_buffer : !vm.buffer,
  %lhs_offset : i64,
  %lhs_strides : tuple<i64, i64>,

  %rhs_buffer : !vm.buffer,
  %rhs_offset : i64,
  %rhs_strides : tuple<i64, i64>,

  %out_buffer : !vm.buffer,
  %out_offset : i64,
  %out_strides : tuple<i64, i64>,

  %sizes : tuple<i64, i64>
)

vm.import private @add.2d.i32(
  %lhs_buffer : !vm.buffer,
  %lhs_offset : i64,
  %lhs_strides : tuple<i64, i64>,

  %rhs_buffer : !vm.buffer,
  %rhs_offset : i64,
  %rhs_strides : tuple<i64, i64>,

  %out_buffer : !vm.buffer,
  %out_offset : i64,
  %out_strides : tuple<i64, i64>,

  %sizes : tuple<i64, i64>
)

vm.import private @and.2d.i32(
  %lhs_buffer : !vm.buffer,
  %lhs_offset : i64,
  %lhs_strides : tuple<i64, i64>,

  %rhs_buffer : !vm.buffer,
  %rhs_offset : i64,
  %rhs_strides : tuple<i64, i64>,

  %out_buffer : !vm.buffer,
  %out_offset : i64,
  %out_strides : tuple<i64, i64>,

  %sizes : tuple<i64, i64>
)

vm.import private @div.2d.f32(
  %lhs_buffer : !vm.buffer,
  %lhs_offset : i64,
  %lhs_strides : tuple<i64, i64>,

  %rhs_buffer : !vm.buffer,
  %rhs_offset : i64,
  %rhs_strides : tuple<i64, i64>,

  %out_buffer : !vm.buffer,
  %out_offset : i64,
  %out_strides : tuple<i64, i64>,

  %sizes : tuple<i64, i64>
)

vm.import private @divs.2d.i32(
  %lhs_buffer : !vm.buffer,
  %lhs_offset : i64,
  %lhs_strides : tuple<i64, i64>,

  %rhs_buffer : !vm.buffer,
  %rhs_offset : i64,
  %rhs_strides : tuple<i64, i64>,

  %out_buffer : !vm.buffer,
  %out_offset : i64,
  %out_strides : tuple<i64, i64>,

  %sizes : tuple<i64, i64>
)

vm.import private @divu.2d.i32(
  %lhs_buffer : !vm.buffer,
  %lhs_offset : i64,
  %lhs_strides : tuple<i64, i64>,

  %rhs_buffer : !vm.buffer,
  %rhs_offset : i64,
  %rhs_strides : tuple<i64, i64>,

  %out_buffer : !vm.buffer,
  %out_offset : i64,
  %out_strides : tuple<i64, i64>,

  %sizes : tuple<i64, i64>
)

vm.import private @mul.2d.f32(
  %lhs_buffer : !vm.buffer,
  %lhs_offset : i64,
  %lhs_strides : tuple<i64, i64>,

  %rhs_buffer : !vm.buffer,
  %rhs_offset : i64,
  %rhs_strides : tuple<i64, i64>,

  %out_buffer : !vm.buffer,
  %out_offset : i64,
  %out_strides : tuple<i64, i64>,

  %sizes : tuple<i64, i64>
)

vm.import private @mul.2d.i32(
  %lhs_buffer : !vm.buffer,
  %lhs_offset : i64,
  %lhs_strides : tuple<i64, i64>,

  %rhs_buffer : !vm.buffer,
  %rhs_offset : i64,
  %rhs_strides : tuple<i64, i64>,

  %out_buffer : !vm.buffer,
  %out_offset : i64,
  %out_strides : tuple<i64, i64>,

  %sizes : tuple<i64, i64>
)

vm.import private @or.2d.i32(
  %lhs_buffer : !vm.buffer,
  %lhs_offset : i64,
  %lhs_strides : tuple<i64, i64>,

  %rhs_buffer : !vm.buffer,
  %rhs_offset : i64,
  %rhs_strides : tuple<i64, i64>,

  %out_buffer : !vm.buffer,
  %out_offset : i64,
  %out_strides : tuple<i64, i64>,

  %sizes : tuple<i64, i64>
)

vm.import private @shl.2d.i32(
  %lhs_buffer : !vm.buffer,
  %lhs_offset : i64,
  %lhs_strides : tuple<i64, i64>,

  %rhs_buffer : !vm.buffer,
  %rhs_offset : i64,
  %rhs_strides : tuple<i64, i64>,

  %out_buffer : !vm.buffer,
  %out_offset : i64,
  %out_strides : tuple<i64, i64>,

  %sizes : tuple<i64, i64>
)

vm.import private @shrs.2d.i32(
  %lhs_buffer : !vm.buffer,
  %lhs_offset : i64,
  %lhs_strides : tuple<i64, i64>,

  %rhs_buffer : !vm.buffer,
  %rhs_offset : i64,
  %rhs_strides : tuple<i64, i64>,

  %out_buffer : !vm.buffer,
  %out_offset : i64,
  %out_strides : tuple<i64, i64>,

  %sizes : tuple<i64, i64>
)

vm.import private @shru.2d.i32(
  %lhs_buffer : !vm.buffer,
  %lhs_offset : i64,
  %lhs_strides : tuple<i64, i64>,

  %rhs_buffer : !vm.buffer,
  %rhs_offset : i64,
  %rhs_strides : tuple<i64, i64>,

  %out_buffer : !vm.buffer,
  %out_offset : i64,
  %out_strides : tuple<i64, i64>,

  %sizes : tuple<i64, i64>
)

vm.import private @sub.2d.f32(
  %lhs_buffer : !vm.buffer,
  %lhs_offset : i64,
  %lhs_strides : tuple<i64, i64>,

  %rhs_buffer : !vm.buffer,
  %rhs_offset : i64,
  %rhs_strides : tuple<i64, i64>,

  %out_buffer : !vm.buffer,
  %out_offset : i64,
  %out_strides : tuple<i64, i64>,

  %sizes : tuple<i64, i64>
)

vm.import private @sub.2d.i32(
  %lhs_buffer : !vm.buffer,
  %lhs_offset : i64,
  %lhs_strides : tuple<i64, i64>,

  %rhs_buffer : !vm.buffer,
  %rhs_offset : i64,
  %rhs_strides : tuple<i64, i64>,

  %out_buffer : !vm.buffer,
  %out_offset : i64,
  %out_strides : tuple<i64, i64>,

  %sizes : tuple<i64, i64>
)

vm.import private @xor.2d.i32(
  %lhs_buffer : !vm.buffer,
  %lhs_offset : i64,
  %lhs_strides : tuple<i64, i64>,

  %rhs_buffer : !vm.buffer,
  %rhs_offset : i64,
  %rhs_strides : tuple<i64, i64>,

  %out_buffer : !vm.buffer,
  %out_offset : i64,
  %out_strides : tuple<i64, i64>,

  %sizes : tuple<i64, i64>
)

//===----------------------------------------------------------------------===//
// VMVX Unary Elementwise Kernels
// Each is specialized by opcode, rank and type width.
//===----------------------------------------------------------------------===//

vm.import private @abs.2d.f32(
  %in_buffer : !vm.buffer,
  %in_offset : i64,
  %in_strides : tuple<i64, i64>,
  %out_buffer : !vm.buffer,
  %out_offset : i64,
  %out_strides : tuple<i64, i64>,
  %sizes : tuple<i64, i64>
)

vm.import private @ceil.2d.f32(
  %in_buffer : !vm.buffer,
  %in_offset : i64,
  %in_strides : tuple<i64, i64>,
  %out_buffer : !vm.buffer,
  %out_offset : i64,
  %out_strides : tuple<i64, i64>,
  %sizes : tuple<i64, i64>
)

vm.import private @ctlz.2d.i32(
  %in_buffer : !vm.buffer,
  %in_offset : i64,
  %in_strides : tuple<i64, i64>,
  %out_buffer : !vm.buffer,
  %out_offset : i64,
  %out_strides : tuple<i64, i64>,
  %sizes : tuple<i64, i64>
)

vm.import private @exp.2d.f32(
  %in_buffer : !vm.buffer,
  %in_offset : i64,
  %in_strides : tuple<i64, i64>,
  %out_buffer : !vm.buffer,
  %out_offset : i64,
  %out_strides : tuple<i64, i64>,
  %sizes : tuple<i64, i64>
)

vm.import private @floor.2d.f32(
  %in_buffer : !vm.buffer,
  %in_offset : i64,
  %in_strides : tuple<i64, i64>,
  %out_buffer : !vm.buffer,
  %out_offset : i64,
  %out_strides : tuple<i64, i64>,
  %sizes : tuple<i64, i64>
)

vm.import private @log.2d.f32(
  %in_buffer : !vm.buffer,
  %in_offset : i64,
  %in_strides : tuple<i64, i64>,
  %out_buffer : !vm.buffer,
  %out_offset : i64,
  %out_strides : tuple<i64, i64>,
  %sizes : tuple<i64, i64>
)

vm.import private @neg.2d.f32(
  %in_buffer : !vm.buffer,
  %in_offset : i64,
  %in_strides : tuple<i64, i64>,
  %out_buffer : !vm.buffer,
  %out_offset : i64,
  %out_strides : tuple<i64, i64>,
  %sizes : tuple<i64, i64>
)

vm.import private @rsqrt.2d.f32(
  %in_buffer : !vm.buffer,
  %in_offset : i64,
  %in_strides : tuple<i64, i64>,
  %out_buffer : !vm.buffer,
  %out_offset : i64,
  %out_strides : tuple<i64, i64>,
  %sizes : tuple<i64, i64>
)

//==============================================================================
// Strided copy ops
// Variants of copy ops exist for power of two rank and datatype sizes.
// Current max rank is 2d.
//==============================================================================
vm.import private @copy.2d.x8(
  %in_buffer : !vm.buffer,
  %in_offset : i64,
  %in_strides : tuple<i64, i64>,
  %out_buffer : !vm.buffer,
  %out_offset : i64,
  %out_strides : tuple<i64, i64>,
  %sizes : tuple<i64, i64>
)

vm.import private @copy.2d.x16(
  %in_buffer : !vm.buffer,
  %in_offset : i64,
  %in_strides : tuple<i64, i64>,
  %out_buffer : !vm.buffer,
  %out_offset : i64,
  %out_strides : tuple<i64, i64>,
  %sizes : tuple<i64, i64>
)

vm.import private @copy.2d.x32(
  %in_buffer : !vm.buffer,
  %in_offset : i64,
  %in_strides : tuple<i64, i64>,
  %out_buffer : !vm.buffer,
  %out_offset : i64,
  %out_strides : tuple<i64, i64>,
  %sizes : tuple<i64, i64>
)

vm.import private @copy.2d.x64(
  %in_buffer : !vm.buffer,
  %in_offset : i64,
  %in_strides : tuple<i64, i64>,
  %out_buffer : !vm.buffer,
  %out_offset : i64,
  %out_strides : tuple<i64, i64>,
  %sizes : tuple<i64, i64>
)

//==============================================================================
// Strided fill ops
//==============================================================================

vm.import private @fill.2d.x32(
  %fill_value : i32,
  %out_buffer : !vm.buffer,
  %out_offset : i64,
  %out_row_stride : i64,
  %size_m : i64,
  %size_n : i64
)

//==============================================================================
// mmt4d ops
//==============================================================================

vm.import private @mmt4d(
  %lhs_buffer : !vm.buffer,
  %lhs_offset : i64,
  %lhs_row_stride : i64,
  %rhs_buffer : !vm.buffer,
  %rhs_offset : i64,
  %rhs_row_stride : i64,
  %out_buffer : !vm.buffer,
  %out_offset : i64,
  %out_row_stride : i64,
  %m : i64,
  %n : i64,
  %k : i64,
  %m0 : i32,
  %n0 : i32,
  %k0 : i32,
  %flags : i32
)

//==============================================================================
// pack ops
//==============================================================================

vm.import private @pack(
  %in_buffer : !vm.buffer,
  %in_offset : i64,
  %in_stride0 : i64,
  %out_buffer : !vm.buffer,
  %out_offset : i64,
  %out_stride0 : i64,
  %in_size0 : i64,
  %in_size1 : i64,
  %out_size0 : i64,
  %out_size1 : i64,
  %out_size2 : i64,
  %out_size3 : i64,
  %padding_value : i64,
  %flags : i32
)

//==============================================================================
// unpack ops
//==============================================================================

vm.import private @unpack(
  %in_buffer : !vm.buffer,
  %in_offset : i64,
  %in_stride0 : i64,
  %out_buffer : !vm.buffer,
  %out_offset : i64,
  %out_stride0 : i64,
  %in_size0 : i64,
  %in_size1 : i64,
  %in_size2 : i64,
  %in_size3 : i64,
  %out_size0 : i64,
  %out_size1 : i64,
  %flags : i32
)

//==============================================================================
// query_tile_size ops
//==============================================================================

vm.import private @query_tile_sizes.2d(
  %sizes : tuple<i64, i64>,
  %flags : i32
) -> (i64, i64)

}  // module

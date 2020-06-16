// IREE VMLA (Virtual Machine-based Linear Algebra) runtime module imports.
//
// This is embedded in the compiler binary and inserted into any module
// containing VMLA dialect ops (vmla.*) that is lowered to the VM dialect.
//
// Element types are embedded in the function. The convention used:
// * 'x': don't-care, bit-depth only.
// * 'i': signed integer
// * 'u': unsigned integer
// * 'f': IREE float
//
// The native module does not need shapes in many cases and only ops that
// actually use the shape information take it as arguments.
//
// When adding methods try to first reuse existing ones. For example, unrolling
// a memcpy to a sequence of vmla.buffer.copy calls (or a loop of them) instead
// of adding a my_batch_copy method.
vm.module @vmla {

//===----------------------------------------------------------------------===//
// VMLA Ops: ABI
//===----------------------------------------------------------------------===//

vm.import @interface.const(
  %interface : !vm.ref<!vmla.interface>,
  %offset : i32
) -> i32
attributes {nosideeffects}

vm.import @interface.binding(
  %interface : !vm.ref<!vmla.interface>,
  %set : i32,
  %binding : i32
) -> !vm.ref<!vmla.buffer>
attributes {nosideeffects}

//===----------------------------------------------------------------------===//
// VMLA Ops: buffer manipulation
//===----------------------------------------------------------------------===//

vm.import @buffer.const(
  %value : !vm.ref<!iree.byte_buffer>
) -> !vm.ref<!vmla.buffer>
attributes {nosideeffects}

vm.import @buffer.alloc(
  %byte_length : i32
) -> !vm.ref<!vmla.buffer>
attributes {nosideeffects}

vm.import @buffer.clone(
  %src : !vm.ref<!vmla.buffer>
) -> !vm.ref<!vmla.buffer>
attributes {nosideeffects}

vm.import @buffer.byte_length(
  %value : !vm.ref<!vmla.buffer>
) -> i32
attributes {nosideeffects}

vm.import @buffer.view(
  %src : !vm.ref<!vmla.buffer>,
  %byte_offset : i32,
  %byte_length : i32
) -> !vm.ref<!vmla.buffer>
attributes {nosideeffects}

vm.import @buffer.copy(
  %src : !vm.ref<!vmla.buffer>, %src_byte_offset : i32,
  %dst : !vm.ref<!vmla.buffer>, %dst_byte_offset : i32,
  %byte_length : i32
)

vm.import @buffer.fill(
  %value : !vm.ref<!vmla.buffer>,
  %dst : !vm.ref<!vmla.buffer>
)

vm.import @buffer.load.i32(
  %src : !vm.ref<!vmla.buffer>,
  %byte_offset : i32
) -> i32
attributes {nosideeffects}

//===----------------------------------------------------------------------===//
// VMLA Ops: comparison
//===----------------------------------------------------------------------===//

vm.import @cmp.i8(%predicate : i32, %lhs : !vm.ref<!vmla.buffer>, %rhs : !vm.ref<!vmla.buffer>, %dst : !vm.ref<!vmla.buffer>)
vm.import @cmp.i16(%predicate : i32, %lhs : !vm.ref<!vmla.buffer>, %rhs : !vm.ref<!vmla.buffer>, %dst : !vm.ref<!vmla.buffer>)
vm.import @cmp.i32(%predicate : i32, %lhs : !vm.ref<!vmla.buffer>, %rhs : !vm.ref<!vmla.buffer>, %dst : !vm.ref<!vmla.buffer>)
vm.import @cmp.f32(%predicate : i32, %lhs : !vm.ref<!vmla.buffer>, %rhs : !vm.ref<!vmla.buffer>, %dst : !vm.ref<!vmla.buffer>)

vm.import @select.x8(%cond : !vm.ref<!vmla.buffer>, %lhs : !vm.ref<!vmla.buffer>, %rhs : !vm.ref<!vmla.buffer>, %dst : !vm.ref<!vmla.buffer>)
vm.import @select.x16(%cond : !vm.ref<!vmla.buffer>, %lhs : !vm.ref<!vmla.buffer>, %rhs : !vm.ref<!vmla.buffer>, %dst : !vm.ref<!vmla.buffer>)
vm.import @select.x32(%cond : !vm.ref<!vmla.buffer>, %lhs : !vm.ref<!vmla.buffer>, %rhs : !vm.ref<!vmla.buffer>, %dst : !vm.ref<!vmla.buffer>)

//===----------------------------------------------------------------------===//
// VMLA Ops: shape/structure
//===----------------------------------------------------------------------===//

// TODO(benvanik): do the copies with buffer.copy instead and leave the offset
// calculations in the IR for the compiler to simplify.
vm.import @copy.x8(
  %src : !vm.ref<!vmla.buffer>, %src_shape : i32 ..., %src_indices : i32 ...,
  %dst : !vm.ref<!vmla.buffer>, %dst_shape : i32 ..., %dst_indices : i32 ...,
  %lengths : i32 ...
)
vm.import @copy.x16(
  %src : !vm.ref<!vmla.buffer>, %src_shape : i32 ..., %src_indices : i32 ...,
  %dst : !vm.ref<!vmla.buffer>, %dst_shape : i32 ..., %dst_indices : i32 ...,
  %lengths : i32 ...
)
vm.import @copy.x32(
  %src : !vm.ref<!vmla.buffer>, %src_shape : i32 ..., %src_indices : i32 ...,
  %dst : !vm.ref<!vmla.buffer>, %dst_shape : i32 ..., %dst_indices : i32 ...,
  %lengths : i32 ...
)

vm.import @transpose.x8(
  %src : !vm.ref<!vmla.buffer>, %src_shape : i32 ...,
  %permutation : i32 ...,
  %dst : !vm.ref<!vmla.buffer>, %dst_shape : i32 ...
)
vm.import @transpose.x16(
  %src : !vm.ref<!vmla.buffer>, %src_shape : i32 ...,
  %permutation : i32 ...,
  %dst : !vm.ref<!vmla.buffer>, %dst_shape : i32 ...
)
vm.import @transpose.x32(
  %src : !vm.ref<!vmla.buffer>, %src_shape : i32 ...,
  %permutation : i32 ...,
  %dst : !vm.ref<!vmla.buffer>, %dst_shape : i32 ...
)

vm.import @reverse.x8(
  %src : !vm.ref<!vmla.buffer>, %src_shape : i32 ...,
  %dimensions : i32 ...,
  %dst : !vm.ref<!vmla.buffer>, %dst_shape : i32 ...
)
vm.import @reverse.x16(
  %src : !vm.ref<!vmla.buffer>, %src_shape : i32 ...,
  %dimensions : i32 ...,
  %dst : !vm.ref<!vmla.buffer>, %dst_shape : i32 ...
)
vm.import @reverse.x32(
  %src : !vm.ref<!vmla.buffer>, %src_shape : i32 ...,
  %dimensions : i32 ...,
  %dst : !vm.ref<!vmla.buffer>, %dst_shape : i32 ...
)

vm.import @pad.x8(
  %src : !vm.ref<!vmla.buffer>, %src_shape : i32 ...,
  %value : !vm.ref<!vmla.buffer>, %value_shape : i32 ...,
  %dst : !vm.ref<!vmla.buffer>, %dst_shape : i32 ...,
  %edge_padding_low : i32 ...,
  %edge_padding_high : i32 ...,
  %interior_padding : i32 ...
)
vm.import @pad.x16(
  %src : !vm.ref<!vmla.buffer>, %src_shape : i32 ...,
  %value : !vm.ref<!vmla.buffer>, %value_shape : i32 ...,
  %dst : !vm.ref<!vmla.buffer>, %dst_shape : i32 ...,
  %edge_padding_low : i32 ..., %edge_padding_high : i32 ...,
  %interior_padding : i32 ...
)
vm.import @pad.x32(
  %src : !vm.ref<!vmla.buffer>, %src_shape : i32 ...,
  %value : !vm.ref<!vmla.buffer>, %value_shape : i32 ...,
  %dst : !vm.ref<!vmla.buffer>, %dst_shape : i32 ...,
  %edge_padding_low : i32 ..., %edge_padding_high : i32 ...,
  %interior_padding : i32 ...
)
vm.import @gather.x8(
  %src : !vm.ref<!vmla.buffer>, %src_shape : i32 ...,
  %indices : !vm.ref<!vmla.buffer>, %indices_shape : i32 ...,
  %dst : !vm.ref<!vmla.buffer>, %dst_shape : i32 ...,
  %dim : i32, %batch_dims : i32
)
vm.import @gather.x16(
  %src : !vm.ref<!vmla.buffer>, %src_shape : i32 ...,
  %indices : !vm.ref<!vmla.buffer>, %indices_shape : i32 ...,
  %dst : !vm.ref<!vmla.buffer>, %dst_shape : i32 ...,
  %dim : i32, %batch_dims : i32
)
vm.import @gather.x32(
  %src : !vm.ref<!vmla.buffer>, %src_shape : i32 ...,
  %indices : !vm.ref<!vmla.buffer>, %indices_shape : i32 ...,
  %dst : !vm.ref<!vmla.buffer>, %dst_shape : i32 ...,
  %dim : i32, %batch_dims : i32
)
vm.import @broadcast.x8(
  %src : !vm.ref<!vmla.buffer>, %src_shape : i32 ...,
  %dst : !vm.ref<!vmla.buffer>, %dst_shape : i32 ...
)
vm.import @broadcast.x16(
  %src : !vm.ref<!vmla.buffer>, %src_shape : i32 ...,
  %dst : !vm.ref<!vmla.buffer>, %dst_shape : i32 ...
)
vm.import @broadcast.x32(
  %src : !vm.ref<!vmla.buffer>, %src_shape : i32 ...,
  %dst : !vm.ref<!vmla.buffer>, %dst_shape : i32 ...
)

vm.import @tile.x8(
  %src : !vm.ref<!vmla.buffer>, %src_shape : i32 ...,
  %dst : !vm.ref<!vmla.buffer>, %dst_shape : i32 ...
)
vm.import @tile.x16(
  %src : !vm.ref<!vmla.buffer>, %src_shape : i32 ...,
  %dst : !vm.ref<!vmla.buffer>, %dst_shape : i32 ...
)
vm.import @tile.x32(
  %src : !vm.ref<!vmla.buffer>, %src_shape : i32 ...,
  %dst : !vm.ref<!vmla.buffer>, %dst_shape : i32 ...
)

//===----------------------------------------------------------------------===//
// VMLA Ops: bit manipulation
//===----------------------------------------------------------------------===//

vm.import @not.x8(%src : !vm.ref<!vmla.buffer>, %dst : !vm.ref<!vmla.buffer>)
vm.import @not.x16(%src : !vm.ref<!vmla.buffer>, %dst : !vm.ref<!vmla.buffer>)
vm.import @not.x32(%src : !vm.ref<!vmla.buffer>, %dst : !vm.ref<!vmla.buffer>)
vm.import @and.x8(%lhs : !vm.ref<!vmla.buffer>, %rhs : !vm.ref<!vmla.buffer>, %dst : !vm.ref<!vmla.buffer>)
vm.import @and.x16(%lhs : !vm.ref<!vmla.buffer>, %rhs : !vm.ref<!vmla.buffer>, %dst : !vm.ref<!vmla.buffer>)
vm.import @and.x32(%lhs : !vm.ref<!vmla.buffer>, %rhs : !vm.ref<!vmla.buffer>, %dst : !vm.ref<!vmla.buffer>)
vm.import @or.x8(%lhs : !vm.ref<!vmla.buffer>, %rhs : !vm.ref<!vmla.buffer>, %dst : !vm.ref<!vmla.buffer>)
vm.import @or.x16(%lhs : !vm.ref<!vmla.buffer>, %rhs : !vm.ref<!vmla.buffer>, %dst : !vm.ref<!vmla.buffer>)
vm.import @or.x32(%lhs : !vm.ref<!vmla.buffer>, %rhs : !vm.ref<!vmla.buffer>, %dst : !vm.ref<!vmla.buffer>)
vm.import @xor.x8(%lhs : !vm.ref<!vmla.buffer>, %rhs : !vm.ref<!vmla.buffer>, %dst : !vm.ref<!vmla.buffer>)
vm.import @xor.x16(%lhs : !vm.ref<!vmla.buffer>, %rhs : !vm.ref<!vmla.buffer>, %dst : !vm.ref<!vmla.buffer>)
vm.import @xor.x32(%lhs : !vm.ref<!vmla.buffer>, %rhs : !vm.ref<!vmla.buffer>, %dst : !vm.ref<!vmla.buffer>)
vm.import @shl.x8(%lhs : !vm.ref<!vmla.buffer>, %rhs : !vm.ref<!vmla.buffer>, %dst : !vm.ref<!vmla.buffer>)
vm.import @shl.x16(%lhs : !vm.ref<!vmla.buffer>, %rhs : !vm.ref<!vmla.buffer>, %dst : !vm.ref<!vmla.buffer>)
vm.import @shl.x32(%lhs : !vm.ref<!vmla.buffer>, %rhs : !vm.ref<!vmla.buffer>, %dst : !vm.ref<!vmla.buffer>)
vm.import @shr.u8(%lhs : !vm.ref<!vmla.buffer>, %rhs : !vm.ref<!vmla.buffer>, %dst : !vm.ref<!vmla.buffer>)
vm.import @shr.u16(%lhs : !vm.ref<!vmla.buffer>, %rhs : !vm.ref<!vmla.buffer>, %dst : !vm.ref<!vmla.buffer>)
vm.import @shr.u32(%lhs : !vm.ref<!vmla.buffer>, %rhs : !vm.ref<!vmla.buffer>, %dst : !vm.ref<!vmla.buffer>)
vm.import @shr.i8(%lhs : !vm.ref<!vmla.buffer>, %rhs : !vm.ref<!vmla.buffer>, %dst : !vm.ref<!vmla.buffer>)
vm.import @shr.i16(%lhs : !vm.ref<!vmla.buffer>, %rhs : !vm.ref<!vmla.buffer>, %dst : !vm.ref<!vmla.buffer>)
vm.import @shr.i32(%lhs : !vm.ref<!vmla.buffer>, %rhs : !vm.ref<!vmla.buffer>, %dst : !vm.ref<!vmla.buffer>)

//===----------------------------------------------------------------------===//
// VMLA Ops: arithmetic
//===----------------------------------------------------------------------===//

vm.import @add.i8(%lhs : !vm.ref<!vmla.buffer>, %rhs : !vm.ref<!vmla.buffer>, %dst : !vm.ref<!vmla.buffer>)
vm.import @add.i16(%lhs : !vm.ref<!vmla.buffer>, %rhs : !vm.ref<!vmla.buffer>, %dst : !vm.ref<!vmla.buffer>)
vm.import @add.i32(%lhs : !vm.ref<!vmla.buffer>, %rhs : !vm.ref<!vmla.buffer>, %dst : !vm.ref<!vmla.buffer>)
vm.import @add.f32(%lhs : !vm.ref<!vmla.buffer>, %rhs : !vm.ref<!vmla.buffer>, %dst : !vm.ref<!vmla.buffer>)
vm.import @sub.i8(%lhs : !vm.ref<!vmla.buffer>, %rhs : !vm.ref<!vmla.buffer>, %dst : !vm.ref<!vmla.buffer>)
vm.import @sub.i16(%lhs : !vm.ref<!vmla.buffer>, %rhs : !vm.ref<!vmla.buffer>, %dst : !vm.ref<!vmla.buffer>)
vm.import @sub.i32(%lhs : !vm.ref<!vmla.buffer>, %rhs : !vm.ref<!vmla.buffer>, %dst : !vm.ref<!vmla.buffer>)
vm.import @sub.f32(%lhs : !vm.ref<!vmla.buffer>, %rhs : !vm.ref<!vmla.buffer>, %dst : !vm.ref<!vmla.buffer>)
vm.import @abs.i8(%src : !vm.ref<!vmla.buffer>, %dst : !vm.ref<!vmla.buffer>)
vm.import @abs.i16(%src : !vm.ref<!vmla.buffer>, %dst : !vm.ref<!vmla.buffer>)
vm.import @abs.i32(%src : !vm.ref<!vmla.buffer>, %dst : !vm.ref<!vmla.buffer>)
vm.import @abs.f32(%src : !vm.ref<!vmla.buffer>, %dst : !vm.ref<!vmla.buffer>)
vm.import @neg.i8(%src : !vm.ref<!vmla.buffer>, %dst : !vm.ref<!vmla.buffer>)
vm.import @neg.i16(%src : !vm.ref<!vmla.buffer>, %dst : !vm.ref<!vmla.buffer>)
vm.import @neg.i32(%src : !vm.ref<!vmla.buffer>, %dst : !vm.ref<!vmla.buffer>)
vm.import @neg.f32(%src : !vm.ref<!vmla.buffer>, %dst : !vm.ref<!vmla.buffer>)
vm.import @mul.i8(%lhs : !vm.ref<!vmla.buffer>, %rhs : !vm.ref<!vmla.buffer>, %dst : !vm.ref<!vmla.buffer>)
vm.import @mul.i16(%lhs : !vm.ref<!vmla.buffer>, %rhs : !vm.ref<!vmla.buffer>, %dst : !vm.ref<!vmla.buffer>)
vm.import @mul.i32(%lhs : !vm.ref<!vmla.buffer>, %rhs : !vm.ref<!vmla.buffer>, %dst : !vm.ref<!vmla.buffer>)
vm.import @mul.f32(%lhs : !vm.ref<!vmla.buffer>, %rhs : !vm.ref<!vmla.buffer>, %dst : !vm.ref<!vmla.buffer>)
vm.import @div.i8(%lhs : !vm.ref<!vmla.buffer>, %rhs : !vm.ref<!vmla.buffer>, %dst : !vm.ref<!vmla.buffer>)
vm.import @div.i16(%lhs : !vm.ref<!vmla.buffer>, %rhs : !vm.ref<!vmla.buffer>, %dst : !vm.ref<!vmla.buffer>)
vm.import @div.i32(%lhs : !vm.ref<!vmla.buffer>, %rhs : !vm.ref<!vmla.buffer>, %dst : !vm.ref<!vmla.buffer>)
vm.import @div.u8(%lhs : !vm.ref<!vmla.buffer>, %rhs : !vm.ref<!vmla.buffer>, %dst : !vm.ref<!vmla.buffer>)
vm.import @div.u16(%lhs : !vm.ref<!vmla.buffer>, %rhs : !vm.ref<!vmla.buffer>, %dst : !vm.ref<!vmla.buffer>)
vm.import @div.u32(%lhs : !vm.ref<!vmla.buffer>, %rhs : !vm.ref<!vmla.buffer>, %dst : !vm.ref<!vmla.buffer>)
vm.import @div.f32(%lhs : !vm.ref<!vmla.buffer>, %rhs : !vm.ref<!vmla.buffer>, %dst : !vm.ref<!vmla.buffer>)
vm.import @rem.i8(%lhs : !vm.ref<!vmla.buffer>, %rhs : !vm.ref<!vmla.buffer>, %dst : !vm.ref<!vmla.buffer>)
vm.import @rem.i16(%lhs : !vm.ref<!vmla.buffer>, %rhs : !vm.ref<!vmla.buffer>, %dst : !vm.ref<!vmla.buffer>)
vm.import @rem.i32(%lhs : !vm.ref<!vmla.buffer>, %rhs : !vm.ref<!vmla.buffer>, %dst : !vm.ref<!vmla.buffer>)
vm.import @rem.u8(%lhs : !vm.ref<!vmla.buffer>, %rhs : !vm.ref<!vmla.buffer>, %dst : !vm.ref<!vmla.buffer>)
vm.import @rem.u16(%lhs : !vm.ref<!vmla.buffer>, %rhs : !vm.ref<!vmla.buffer>, %dst : !vm.ref<!vmla.buffer>)
vm.import @rem.u32(%lhs : !vm.ref<!vmla.buffer>, %rhs : !vm.ref<!vmla.buffer>, %dst : !vm.ref<!vmla.buffer>)
vm.import @rem.f32(%lhs : !vm.ref<!vmla.buffer>, %rhs : !vm.ref<!vmla.buffer>, %dst : !vm.ref<!vmla.buffer>)
vm.import @pow.f32(%lhs : !vm.ref<!vmla.buffer>, %rhs : !vm.ref<!vmla.buffer>, %dst : !vm.ref<!vmla.buffer>)
vm.import @exp.f32(%src : !vm.ref<!vmla.buffer>, %dst : !vm.ref<!vmla.buffer>)
vm.import @log.f32(%src : !vm.ref<!vmla.buffer>, %dst : !vm.ref<!vmla.buffer>)
vm.import @rsqrt.f32(%src : !vm.ref<!vmla.buffer>, %dst : !vm.ref<!vmla.buffer>)
vm.import @sqrt.f32(%src : !vm.ref<!vmla.buffer>, %dst : !vm.ref<!vmla.buffer>)
vm.import @cos.f32(%src : !vm.ref<!vmla.buffer>, %dst : !vm.ref<!vmla.buffer>)
vm.import @sin.f32(%src : !vm.ref<!vmla.buffer>, %dst : !vm.ref<!vmla.buffer>)
vm.import @tanh.f32(%src : !vm.ref<!vmla.buffer>, %dst : !vm.ref<!vmla.buffer>)
vm.import @atan2.f32(%lhs : !vm.ref<!vmla.buffer>, %rhs : !vm.ref<!vmla.buffer>, %dst : !vm.ref<!vmla.buffer>)

vm.import @min.i8(%lhs : !vm.ref<!vmla.buffer>, %rhs : !vm.ref<!vmla.buffer>, %dst : !vm.ref<!vmla.buffer>)
vm.import @min.i16(%lhs : !vm.ref<!vmla.buffer>, %rhs : !vm.ref<!vmla.buffer>, %dst : !vm.ref<!vmla.buffer>)
vm.import @min.i32(%lhs : !vm.ref<!vmla.buffer>, %rhs : !vm.ref<!vmla.buffer>, %dst : !vm.ref<!vmla.buffer>)
vm.import @min.f32(%lhs : !vm.ref<!vmla.buffer>, %rhs : !vm.ref<!vmla.buffer>, %dst : !vm.ref<!vmla.buffer>)
vm.import @max.i8(%lhs : !vm.ref<!vmla.buffer>, %rhs : !vm.ref<!vmla.buffer>, %dst : !vm.ref<!vmla.buffer>)
vm.import @max.i16(%lhs : !vm.ref<!vmla.buffer>, %rhs : !vm.ref<!vmla.buffer>, %dst : !vm.ref<!vmla.buffer>)
vm.import @max.i32(%lhs : !vm.ref<!vmla.buffer>, %rhs : !vm.ref<!vmla.buffer>, %dst : !vm.ref<!vmla.buffer>)
vm.import @max.f32(%lhs : !vm.ref<!vmla.buffer>, %rhs : !vm.ref<!vmla.buffer>, %dst : !vm.ref<!vmla.buffer>)
vm.import @clamp.i8(%min : !vm.ref<!vmla.buffer>, %value : !vm.ref<!vmla.buffer>, %max : !vm.ref<!vmla.buffer>, %dst : !vm.ref<!vmla.buffer>)
vm.import @clamp.i16(%min : !vm.ref<!vmla.buffer>, %value : !vm.ref<!vmla.buffer>, %max : !vm.ref<!vmla.buffer>, %dst : !vm.ref<!vmla.buffer>)
vm.import @clamp.i32(%min : !vm.ref<!vmla.buffer>, %value : !vm.ref<!vmla.buffer>, %max : !vm.ref<!vmla.buffer>, %dst : !vm.ref<!vmla.buffer>)
vm.import @clamp.f32(%min : !vm.ref<!vmla.buffer>, %value : !vm.ref<!vmla.buffer>, %max : !vm.ref<!vmla.buffer>, %dst : !vm.ref<!vmla.buffer>)
vm.import @floor.f32(%src : !vm.ref<!vmla.buffer>, %dst : !vm.ref<!vmla.buffer>)
vm.import @ceil.f32(%src : !vm.ref<!vmla.buffer>, %dst : !vm.ref<!vmla.buffer>)

//===----------------------------------------------------------------------===//
// VMLA Ops: conversion
//===----------------------------------------------------------------------===//

vm.import @convert.i8.i16(%src : !vm.ref<!vmla.buffer>, %dst : !vm.ref<!vmla.buffer>)
vm.import @convert.i8.i32(%src : !vm.ref<!vmla.buffer>, %dst : !vm.ref<!vmla.buffer>)
vm.import @convert.i8.f32(%src : !vm.ref<!vmla.buffer>, %dst : !vm.ref<!vmla.buffer>)
vm.import @convert.i16.i8(%src : !vm.ref<!vmla.buffer>, %dst : !vm.ref<!vmla.buffer>)
vm.import @convert.i16.i32(%src : !vm.ref<!vmla.buffer>, %dst : !vm.ref<!vmla.buffer>)
vm.import @convert.i16.f32(%src : !vm.ref<!vmla.buffer>, %dst : !vm.ref<!vmla.buffer>)
vm.import @convert.i32.i8(%src : !vm.ref<!vmla.buffer>, %dst : !vm.ref<!vmla.buffer>)
vm.import @convert.i32.i16(%src : !vm.ref<!vmla.buffer>, %dst : !vm.ref<!vmla.buffer>)
vm.import @convert.i32.f32(%src : !vm.ref<!vmla.buffer>, %dst : !vm.ref<!vmla.buffer>)
vm.import @convert.f32.i8(%src : !vm.ref<!vmla.buffer>, %dst : !vm.ref<!vmla.buffer>)
vm.import @convert.f32.i16(%src : !vm.ref<!vmla.buffer>, %dst : !vm.ref<!vmla.buffer>)
vm.import @convert.f32.i32(%src : !vm.ref<!vmla.buffer>, %dst : !vm.ref<!vmla.buffer>)

//===----------------------------------------------------------------------===//
// VMLA Ops: Convolution
//===----------------------------------------------------------------------===//

vm.import @conv.f32f32.f32(
  %input: !vm.ref<!vmla.buffer>, %input_shape: i32 ...,
  %filter: !vm.ref<!vmla.buffer>, %filter_shape: i32 ...,
  %dst: !vm.ref<!vmla.buffer>, %dst_shape: i32 ...,
  %window_strides: i32 ...,
  %padding: i32 ...,
  %lhs_dilation: i32 ...,
  %rhs_dilation: i32 ...,
  %feature_group_count: i32,
  %batch_group_count: i32
)

//===----------------------------------------------------------------------===//
// VMLA Ops: GEMM/GEMV
//===----------------------------------------------------------------------===//

vm.import @batch.matmul.f32f32.f32(
  %lhs : !vm.ref<!vmla.buffer>, %lhs_shape : i32 ...,
  %rhs : !vm.ref<!vmla.buffer>, %rhs_shape : i32 ...,
  %dst : !vm.ref<!vmla.buffer>, %dst_shape : i32 ...
)

//===----------------------------------------------------------------------===//
// VMLA Ops: reduction
//===----------------------------------------------------------------------===//

vm.import @reduce.sum.i8(
  %src : !vm.ref<!vmla.buffer>, %src_shape : i32 ...,
  %init : !vm.ref<!vmla.buffer>, %init_shape : i32 ...,
  %dimension : i32,
  %dst : !vm.ref<!vmla.buffer>, %dst_shape : i32 ...
)
vm.import @reduce.sum.i16(
  %src : !vm.ref<!vmla.buffer>, %src_shape : i32 ...,
  %init : !vm.ref<!vmla.buffer>, %init_shape : i32 ...,
  %dimension : i32,
  %dst : !vm.ref<!vmla.buffer>, %dst_shape : i32 ...
)
vm.import @reduce.sum.i32(
  %src : !vm.ref<!vmla.buffer>, %src_shape : i32 ...,
  %init : !vm.ref<!vmla.buffer>, %init_shape : i32 ...,
  %dimension : i32,
  %dst : !vm.ref<!vmla.buffer>, %dst_shape : i32 ...
)
vm.import @reduce.sum.f32(
  %src : !vm.ref<!vmla.buffer>, %src_shape : i32 ...,
  %init : !vm.ref<!vmla.buffer>, %init_shape : i32 ...,
  %dimension : i32,
  %dst : !vm.ref<!vmla.buffer>, %dst_shape : i32 ...
)

vm.import @reduce.min.i8(
  %src : !vm.ref<!vmla.buffer>, %src_shape : i32 ...,
  %init : !vm.ref<!vmla.buffer>, %init_shape : i32 ...,
  %dimension : i32,
  %dst : !vm.ref<!vmla.buffer>, %dst_shape : i32 ...
)
vm.import @reduce.min.i16(
  %src : !vm.ref<!vmla.buffer>, %src_shape : i32 ...,
  %init : !vm.ref<!vmla.buffer>, %init_shape : i32 ...,
  %dimension : i32,
  %dst : !vm.ref<!vmla.buffer>, %dst_shape : i32 ...
)
vm.import @reduce.min.i32(
  %src : !vm.ref<!vmla.buffer>, %src_shape : i32 ...,
  %init : !vm.ref<!vmla.buffer>, %init_shape : i32 ...,
  %dimension : i32,
  %dst : !vm.ref<!vmla.buffer>, %dst_shape : i32 ...
)
vm.import @reduce.min.f32(
  %src : !vm.ref<!vmla.buffer>, %src_shape : i32 ...,
  %init : !vm.ref<!vmla.buffer>, %init_shape : i32 ...,
  %dimension : i32,
  %dst : !vm.ref<!vmla.buffer>, %dst_shape : i32 ...
)

vm.import @reduce.max.i8(
  %src : !vm.ref<!vmla.buffer>, %src_shape : i32 ...,
  %init : !vm.ref<!vmla.buffer>, %init_shape : i32 ...,
  %dimension : i32,
  %dst : !vm.ref<!vmla.buffer>, %dst_shape : i32 ...
)
vm.import @reduce.max.i16(
  %src : !vm.ref<!vmla.buffer>, %src_shape : i32 ...,
  %init : !vm.ref<!vmla.buffer>, %init_shape : i32 ...,
  %dimension : i32,
  %dst : !vm.ref<!vmla.buffer>, %dst_shape : i32 ...
)
vm.import @reduce.max.i32(
  %src : !vm.ref<!vmla.buffer>, %src_shape : i32 ...,
  %init : !vm.ref<!vmla.buffer>, %init_shape : i32 ...,
  %dimension : i32,
  %dst : !vm.ref<!vmla.buffer>, %dst_shape : i32 ...
)
vm.import @reduce.max.f32(
  %src : !vm.ref<!vmla.buffer>, %src_shape : i32 ...,
  %init : !vm.ref<!vmla.buffer>, %init_shape : i32 ...,
  %dimension : i32,
  %dst : !vm.ref<!vmla.buffer>, %dst_shape : i32 ...
)

vm.import @pooling.sum.i8(
  %src : !vm.ref<!vmla.buffer>, %src_shape : i32 ...,
  %init : !vm.ref<!vmla.buffer>, %init_shape : i32 ...,
  %dst : !vm.ref<!vmla.buffer>, %dst_shape : i32 ...,
  %window_dimensions: i32 ...,
  %window_strides: i32 ...,
  %padding: i32 ...
)
vm.import @pooling.sum.i16(
  %src : !vm.ref<!vmla.buffer>, %src_shape : i32 ...,
  %init : !vm.ref<!vmla.buffer>, %init_shape : i32 ...,
  %dst : !vm.ref<!vmla.buffer>, %dst_shape : i32 ...,
  %window_dimensions: i32 ...,
  %window_strides: i32 ...,
  %padding: i32 ...
)
vm.import @pooling.sum.i32(
  %src : !vm.ref<!vmla.buffer>, %src_shape : i32 ...,
  %init : !vm.ref<!vmla.buffer>, %init_shape : i32 ...,
  %dst : !vm.ref<!vmla.buffer>, %dst_shape : i32 ...,
  %window_dimensions: i32 ...,
  %window_strides: i32 ...,
  %padding: i32 ...
)
vm.import @pooling.sum.f32(
  %src : !vm.ref<!vmla.buffer>, %src_shape : i32 ...,
  %init : !vm.ref<!vmla.buffer>, %init_shape : i32 ...,
  %dst : !vm.ref<!vmla.buffer>, %dst_shape : i32 ...,
  %window_dimensions: i32 ...,
  %window_strides: i32 ...,
  %padding: i32 ...
)

vm.import @pooling.min.i8(
  %src : !vm.ref<!vmla.buffer>, %src_shape : i32 ...,
  %init : !vm.ref<!vmla.buffer>, %init_shape : i32 ...,
  %dst : !vm.ref<!vmla.buffer>, %dst_shape : i32 ...,
  %window_dimensions: i32 ...,
  %window_strides: i32 ...,
  %padding: i32 ...
)
vm.import @pooling.min.i16(
  %src : !vm.ref<!vmla.buffer>, %src_shape : i32 ...,
  %init : !vm.ref<!vmla.buffer>, %init_shape : i32 ...,
  %dst : !vm.ref<!vmla.buffer>, %dst_shape : i32 ...,
  %window_dimensions: i32 ...,
  %window_strides: i32 ...,
  %padding: i32 ...
)
vm.import @pooling.min.i32(
  %src : !vm.ref<!vmla.buffer>, %src_shape : i32 ...,
  %init : !vm.ref<!vmla.buffer>, %init_shape : i32 ...,
  %dst : !vm.ref<!vmla.buffer>, %dst_shape : i32 ...,
  %window_dimensions: i32 ...,
  %window_strides: i32 ...,
  %padding: i32 ...
)
vm.import @pooling.min.f32(
  %src : !vm.ref<!vmla.buffer>, %src_shape : i32 ...,
  %init : !vm.ref<!vmla.buffer>, %init_shape : i32 ...,
  %dst : !vm.ref<!vmla.buffer>, %dst_shape : i32 ...,
  %window_dimensions: i32 ...,
  %window_strides: i32 ...,
  %padding: i32 ...
)

vm.import @pooling.max.i8(
  %src : !vm.ref<!vmla.buffer>, %src_shape : i32 ...,
  %init : !vm.ref<!vmla.buffer>, %init_shape : i32 ...,
  %dst : !vm.ref<!vmla.buffer>, %dst_shape : i32 ...,
  %window_dimensions: i32 ...,
  %window_strides: i32 ...,
  %padding: i32 ...
)
vm.import @pooling.max.i16(
  %src : !vm.ref<!vmla.buffer>, %src_shape : i32 ...,
  %init : !vm.ref<!vmla.buffer>, %init_shape : i32 ...,
  %dst : !vm.ref<!vmla.buffer>, %dst_shape : i32 ...,
  %window_dimensions: i32 ...,
  %window_strides: i32 ...,
  %padding: i32 ...
)
vm.import @pooling.max.i32(
  %src : !vm.ref<!vmla.buffer>, %src_shape : i32 ...,
  %init : !vm.ref<!vmla.buffer>, %init_shape : i32 ...,
  %dst : !vm.ref<!vmla.buffer>, %dst_shape : i32 ...,
  %window_dimensions: i32 ...,
  %window_strides: i32 ...,
  %padding: i32 ...
)
vm.import @pooling.max.f32(
  %src : !vm.ref<!vmla.buffer>, %src_shape : i32 ...,
  %init : !vm.ref<!vmla.buffer>, %init_shape : i32 ...,
  %dst : !vm.ref<!vmla.buffer>, %dst_shape : i32 ...,
  %window_dimensions: i32 ...,
  %window_strides: i32 ...,
  %padding: i32 ...
)

}  // module

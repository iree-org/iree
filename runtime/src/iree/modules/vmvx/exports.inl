// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===----------------------------------------------------------------------===//
//
//         ██     ██  █████  ██████  ███    ██ ██ ███    ██  ██████
//         ██     ██ ██   ██ ██   ██ ████   ██ ██ ████   ██ ██
//         ██  █  ██ ███████ ██████  ██ ██  ██ ██ ██ ██  ██ ██   ███
//         ██ ███ ██ ██   ██ ██   ██ ██  ██ ██ ██ ██  ██ ██ ██    ██
//          ███ ███  ██   ██ ██   ██ ██   ████ ██ ██   ████  ██████
//
//===----------------------------------------------------------------------===//
//
// This file matches the vmvx.imports.mlir in the compiler. It'd be nice to
// autogenerate this as the order of these functions must be sorted ascending by
// name in a way compatible with iree_string_view_compare.
//
// Users are meant to `#define EXPORT_FN` to be able to access the information.
// #define EXPORT_FN(name, target_fn, arg_struct, arg_type, ret_type)

// clang-format off

EXPORT_FN("abs.2d.f32", iree_uk_x32u_absf_2d, ukernel_x32u_2d, rIrIIIIIII, v)
EXPORT_FN("add.2d.f32", iree_uk_x32b_addf_2d, ukernel_x32b_2d, rIrIrIIIIIIIII, v)
EXPORT_FN("add.2d.i32", iree_uk_x32b_addi_2d, ukernel_x32b_2d, rIrIrIIIIIIIII, v)
EXPORT_FN("and.2d.i32", iree_uk_x32b_andi_2d, ukernel_x32b_2d, rIrIrIIIIIIIII, v)
EXPORT_FN("ceil.2d.f32", iree_uk_x32u_ceilf_2d, ukernel_x32u_2d, rIrIIIIIII, v)
EXPORT_FN("copy.2d.x16", iree_vmvx_copy2d_x16, unary2d, rIrIIIIIII, v)
EXPORT_FN("copy.2d.x32", iree_vmvx_copy2d_x32, unary2d, rIrIIIIIII, v)
EXPORT_FN("copy.2d.x64", iree_vmvx_copy2d_x64, unary2d, rIrIIIIIII, v)
EXPORT_FN("copy.2d.x8", iree_vmvx_copy2d_x8, unary2d, rIrIIIIIII, v)
EXPORT_FN("ctlz.2d.i32", iree_uk_x32u_ctlz_2d, ukernel_x32u_2d, rIrIIIIIII, v)
EXPORT_FN("div.2d.f32", iree_uk_x32b_divf_2d, ukernel_x32b_2d, rIrIrIIIIIIIII, v)
EXPORT_FN("divs.2d.i32", iree_uk_x32b_divsi_2d, ukernel_x32b_2d, rIrIrIIIIIIIII, v)
EXPORT_FN("divu.2d.i32", iree_uk_x32b_divui_2d, ukernel_x32b_2d, rIrIrIIIIIIIII, v)
EXPORT_FN("exp.2d.f32", iree_uk_x32u_expf_2d, ukernel_x32u_2d, rIrIIIIIII, v)
EXPORT_FN("fill.2d.x32", iree_vmvx_fill2d_x32, fill2d_x32, riIIII, v)
EXPORT_FN("floor.2d.f32", iree_uk_x32u_floorf_2d, ukernel_x32u_2d, rIrIIIIIII, v)
EXPORT_FN("log.2d.f32", iree_uk_x32u_logf_2d, ukernel_x32u_2d, rIrIIIIIII, v)
EXPORT_FN("matmul.f32f32f32", iree_vmvx_matmul_f32f32f32, matmul, rIIrIIrIIIIIi, v)
EXPORT_FN("matmul.i8i8i32", iree_vmvx_matmul_i8i8i32, matmul, rIIrIIrIIIIIi, v)
EXPORT_FN("mmt4d.f32f32f32", iree_vmvx_mmt4d_f32f32f32, mmt4d, rIrIrIIIIIIIiiii, v)
EXPORT_FN("mmt4d.i8i8i32", iree_vmvx_mmt4d_i8i8i32, mmt4d, rIrIrIIIIIIIiiii, v)
EXPORT_FN("mul.2d.f32", iree_uk_x32b_mulf_2d, ukernel_x32b_2d, rIrIrIIIIIIIII, v)
EXPORT_FN("mul.2d.i32", iree_uk_x32b_muli_2d, ukernel_x32b_2d, rIrIrIIIIIIIII, v)
EXPORT_FN("neg.2d.f32", iree_uk_x32u_negf_2d, ukernel_x32u_2d, rIrIIIIIII, v)
EXPORT_FN("or.2d.i32", iree_uk_x32b_ori_2d, ukernel_x32b_2d, rIrIrIIIIIIIII, v)
EXPORT_FN("pack.f32f32", iree_vmvx_pack_f32f32, pack_f, rIrIIIIIIIIIfi, v)
EXPORT_FN("pack.i32i32", iree_vmvx_pack_i32i32, pack_i, rIrIIIIIIIIIii, v)
EXPORT_FN("pack.i8i8", iree_vmvx_pack_i8i8, pack_i, rIrIIIIIIIIIii, v)
EXPORT_FN("query_tile_sizes.2d", iree_vmvx_query_tile_sizes_2d, query_tile_sizes_2d, IIi, II)
EXPORT_FN("rsqrt.2d.f32", iree_uk_x32u_rsqrtf_2d, ukernel_x32u_2d, rIrIIIIIII, v)
EXPORT_FN("shl.2d.i32", iree_uk_x32b_shli_2d, ukernel_x32b_2d, rIrIrIIIIIIIII, v)
EXPORT_FN("shrs.2d.i32", iree_uk_x32b_shrsi_2d, ukernel_x32b_2d, rIrIrIIIIIIIII, v)
EXPORT_FN("shru.2d.i32", iree_uk_x32b_shrui_2d, ukernel_x32b_2d, rIrIrIIIIIIIII, v)
EXPORT_FN("sub.2d.f32", iree_uk_x32b_subf_2d, ukernel_x32b_2d, rIrIrIIIIIIIII, v)
EXPORT_FN("sub.2d.i32", iree_uk_x32b_subi_2d, ukernel_x32b_2d, rIrIrIIIIIIIII, v)
EXPORT_FN("unpack.f32f32", iree_vmvx_unpack_f32f32, unpack, rIrIIIIIIIIIi, v)
EXPORT_FN("unpack.i32i32", iree_vmvx_unpack_i32i32, unpack, rIrIIIIIIIIIi, v)
EXPORT_FN("unpack.i8i8", iree_vmvx_unpack_i8i8, unpack, rIrIIIIIIIIIi, v)
EXPORT_FN("xor.2d.i32", iree_uk_x32b_xori_2d, ukernel_x32b_2d, rIrIrIIIIIIIII, v)

// clang-format on

#!/usr/bin/env python3
# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Generator for the paged-KV CUDA e2e attention test.

This follows the structure of `generate_e2e_attention_tests.py`. The batch,
page pool and head dimensions are fixed, while the number of pages per request
is dynamic so that the test exercises the same shape behavior a real model has.
"""

import argparse
import math
import struct

BATCH = 4
NUM_PAGES = 4
POOL_PAGES = 8
PAGE_SIZE = 32
HEAD_DIM = 128

# The attention op consumes an f16 scale, so round 1/sqrt(HEAD_DIM) to f16 here
# and reuse the exact same value for the host reference. Keeping both sides on
# one Python value avoids a silently diverging magic constant.
SCALE = struct.unpack("<e", struct.pack("<e", 1.0 / math.sqrt(HEAD_DIM)))[0]

QUERY_TYPE = f"tensor<{BATCH}x1x1x{HEAD_DIM}xf16>"
KV_STORAGE_TYPE = f"tensor<{POOL_PAGES}x1x{PAGE_SIZE}x{HEAD_DIM}xf16>"
PAGE_TABLE_TYPE = f"tensor<{BATCH}x?xi64>"
GATHERED_KV_TYPE = f"tensor<{BATCH}x?x1x{PAGE_SIZE}x{HEAD_DIM}xf16>"
MASK_TYPE = f"tensor<{BATCH}x1x1x?x{PAGE_SIZE}xf32>"
OUTPUT_TYPE = f"tensor<{BATCH}x1x1x{HEAD_DIM}xf32>"
STATISTICS_TYPE = f"tensor<{BATCH}x1x1xf32>"


def gather_kv_pages(result, page_table, output):
    """Emits a gather of one logical KV tensor out of the physical page pool."""
    return f"""\
  %{result} = iree_linalg_ext.gather dimension_map = [0]
      ins(%kv_storage, %{page_table} : {KV_STORAGE_TYPE}, {PAGE_TABLE_TYPE})
      outs(%{output} : {GATHERED_KV_TYPE})
      -> {GATHERED_KV_TYPE}
"""


def generate_attention_mlir(output_path):
    text = f"""\
func.func @paged_attention(
    %query: {QUERY_TYPE},
    %kv_storage: {KV_STORAGE_TYPE},
    %key_page_table: {PAGE_TABLE_TYPE},
    %value_page_table: {PAGE_TABLE_TYPE})
    -> {OUTPUT_TYPE} {{
  %zero_f32 = arith.constant 0.0 : f32
  %neg_inf_f32 = arith.constant 0xFF800000 : f32
  %scale = arith.constant {SCALE!r} : f16
  %c1 = arith.constant 1 : index
  %num_pages = tensor.dim %key_page_table, %c1 : {PAGE_TABLE_TYPE}
  %key_empty = tensor.empty(%num_pages) : {GATHERED_KV_TYPE}
  %value_empty = tensor.empty(%num_pages) : {GATHERED_KV_TYPE}
{gather_kv_pages("key", "key_page_table", "key_empty")}\
{gather_kv_pages("value", "value_page_table", "value_empty")}\
  // All-zero additive mask. It leaves real scores untouched and only exists so
  // that the compiler is able to pad the dynamic key/value dimension.
  %mask_empty = tensor.empty(%num_pages) : {MASK_TYPE}
  %mask = linalg.fill ins(%zero_f32 : f32)
      outs(%mask_empty : {MASK_TYPE}) -> {MASK_TYPE}
  %output_empty = tensor.empty() : {OUTPUT_TYPE}
  %max_empty = tensor.empty() : {STATISTICS_TYPE}
  %sum_empty = tensor.empty() : {STATISTICS_TYPE}
  %output_init = linalg.fill ins(%zero_f32 : f32)
      outs(%output_empty : {OUTPUT_TYPE}) -> {OUTPUT_TYPE}
  %max_init = linalg.fill ins(%neg_inf_f32 : f32)
      outs(%max_empty : {STATISTICS_TYPE}) -> {STATISTICS_TYPE}
  %sum_init = linalg.fill ins(%zero_f32 : f32)
      outs(%sum_empty : {STATISTICS_TYPE}) -> {STATISTICS_TYPE}
  %attention:3 = iree_linalg_ext.online_attention {{
      indexing_maps = [
        affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1, d2, d4)>,
        affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d5, d1, d6, d4)>,
        affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d5, d1, d6, d3)>,
        affine_map<(d0, d1, d2, d3, d4, d5, d6) -> ()>,
        affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1, d2, d5, d6)>,
        affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1, d2, d3)>,
        affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1, d2)>,
        affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1, d2)>]
    }}
    ins(%query, %key, %value, %scale, %mask :
      {QUERY_TYPE}, {GATHERED_KV_TYPE}, {GATHERED_KV_TYPE}, f16, {MASK_TYPE})
    outs(%output_init, %max_init, %sum_init :
      {OUTPUT_TYPE}, {STATISTICS_TYPE}, {STATISTICS_TYPE}) {{
      ^bb0(%score: f32):
        iree_linalg_ext.yield %score : f32
    }} -> {OUTPUT_TYPE}, {STATISTICS_TYPE}, {STATISTICS_TYPE}
  %normalized_empty = tensor.empty() : {OUTPUT_TYPE}
  %normalized = linalg.generic {{
      indexing_maps = [
        affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
        affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>,
        affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
      iterator_types = ["parallel", "parallel", "parallel", "parallel"]
    }} ins(%attention#0, %attention#2 : {OUTPUT_TYPE}, {STATISTICS_TYPE})
    outs(%normalized_empty : {OUTPUT_TYPE}) {{
      ^bb0(%attn_value: f32, %sum: f32, %out: f32):
        %result = arith.divf %attn_value, %sum : f32
        linalg.yield %result : f32
    }} -> {OUTPUT_TYPE}
  return %normalized : {OUTPUT_TYPE}
}}
"""
    with open(output_path, "w") as file:
        file.write(text)


def page_table_entries(offset, stride):
    """Builds a deterministic, non-contiguous logical-to-physical page mapping.

    The strides are coprime with POOL_PAGES so the mapping walks the whole page
    pool, both strides are odd so the key and value tables never select the same
    physical page for one slot, and the batch index shifts the mapping so no two
    requests share a page sequence.
    """
    return [
        (offset + stride * page + batch) % POOL_PAGES
        for batch in range(BATCH)
        for page in range(NUM_PAGES)
    ]


def emit_page_table(name, pages):
    """Emits one host-constant page table and exports it as a buffer view."""
    flat_type = f"tensor<{BATCH * NUM_PAGES}xi64>"
    table_type = f"tensor<{BATCH}x{NUM_PAGES}xi64>"
    values = ", ".join(str(page) for page in pages)
    return f"""\
    %{name}_flat = arith.constant dense<[{values}]> : {flat_type}
    %{name}_table = tensor.expand_shape %{name}_flat [[0, 1]] output_shape [{BATCH}, {NUM_PAGES}] : {flat_type} into {table_type}
    %{name}_page_table = hal.tensor.export %{name}_table : {table_type} -> !hal.buffer_view
"""


def generate_calls_mlir(output_path):
    key_pages = page_table_entries(3, 5)
    value_pages = page_table_entries(6, 7)
    text = f"""\
builtin.module @calls {{
  func.func private @attention_test.generate_random_4d_tensor(
      %device: !hal.device, %dim0: i64, %dim1: i64, %dim2: i64,
      %dim3: i64, %element_type: i32, %seed: i32) -> !hal.buffer_view
  func.func private @attention_test.check_paged_attention_results(
      %device: !hal.device, %batch: i64, %num_pages: i64, %page_size: i64,
      %head_dim: i64, %scale: f32, %query: !hal.buffer_view,
      %kv_storage: !hal.buffer_view, %key_page_table: !hal.buffer_view,
      %value_page_table: !hal.buffer_view, %result: !hal.buffer_view)
  func.func private @module.paged_attention(
      %query: !hal.buffer_view, %kv_storage: !hal.buffer_view,
      %key_page_table: !hal.buffer_view, %value_page_table: !hal.buffer_view)
      -> !hal.buffer_view
  func.func @paged_attention_test() attributes {{
    iree.reflection = {{description = "Paged KV attention B{BATCH} pages{NUM_PAGES} page{PAGE_SIZE} head{HEAD_DIM} f16"}}
  }} {{
    %device_index = arith.constant 0 : index
    %device = hal.devices.get %device_index : !hal.device
    %batch = arith.constant {BATCH} : i64
    %num_pages = arith.constant {NUM_PAGES} : i64
    %page_size = arith.constant {PAGE_SIZE} : i64
    %head_dim = arith.constant {HEAD_DIM} : i64
    %scale = arith.constant {SCALE!r} : f32
    %c1 = arith.constant 1 : i64
    %pool_pages = arith.constant {POOL_PAGES} : i64
    %query_seed = arith.constant 17 : i32
    %kv_seed = arith.constant 23 : i32
    %query_type = hal.element_type<f16> : i32
    %query = call @attention_test.generate_random_4d_tensor(
      %device, %batch, %c1, %c1, %head_dim, %query_type, %query_seed) : (!hal.device, i64, i64, i64, i64, i32, i32) -> !hal.buffer_view
    %kv = call @attention_test.generate_random_4d_tensor(
      %device, %pool_pages, %c1, %page_size, %head_dim, %query_type, %kv_seed) : (!hal.device, i64, i64, i64, i64, i32, i32) -> !hal.buffer_view
{emit_page_table("key", key_pages)}\
{emit_page_table("value", value_pages)}\
    %result = call @module.paged_attention(%query, %kv, %key_page_table, %value_page_table) : (!hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view) -> !hal.buffer_view
    call @attention_test.check_paged_attention_results(%device, %batch, %num_pages, %page_size, %head_dim, %scale, %query, %kv, %key_page_table, %value_page_table, %result) : (!hal.device, i64, i64, i64, i64, f32, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view) -> ()
    return
  }}
}}
"""
    with open(output_path, "w") as file:
        file.write(text)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_attention_mlir", required=True)
    parser.add_argument("--output_calls_mlir", required=True)
    args = parser.parse_args()
    generate_attention_mlir(args.output_attention_mlir)
    generate_calls_mlir(args.output_calls_mlir)


if __name__ == "__main__":
    main()

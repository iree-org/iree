#!/usr/bin/env python3
# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import argparse

BATCH = 4
NUM_PAGES = 4
POOL_PAGES = 8
PAGE_SIZE = 32
HEAD_DIM = 128


def generate_attention_mlir(output_path):
    text = f"""\
#executable_target_cuda = #hal.executable.target<\"cuda\", \"cuda-nvptx-fb\">
#translation = #iree_codegen.translation_info<
    pipeline = #iree_gpu.pipeline<VectorDistribute>
    workgroup_size = [128, 1, 1]
    subgroup_size = 32>
#qk_attrs_config = #iree_gpu.lowering_config<{{
    subgroup_basis = [[1, 1, 1, 1, 1, 4, 1], [4, 3, 2, 1, 5, 6]],
    thread = [0, 0, 0, 8, 0, 0],
    lane_basis = [[1, 1, 1, 1, 1, 2, 16], [2, 1, 0, 4, 5, 6]]
}}>
#pv_attrs_config = #iree_gpu.lowering_config<{{
    subgroup_basis = [[1, 1, 1, 1, 1, 4, 1], [4, 3, 2, 1, 5, 6]],
    thread = [0, 0, 0, 8, 0, 0],
    lane_basis = [[1, 1, 1, 1, 1, 2, 16], [2, 1, 0, 4, 5, 6]]
}}>
#attention_lowering_config = #iree_gpu.lowering_config<{{
    partial_reduction = [0, 0, 0, 0, 0, 8, 0],
    workgroup = [1, 1, 1, 0, 0, 0, 0]
}}>

func.func @paged_attention(
    %query: tensor<{BATCH}x1x1x{HEAD_DIM}xf16>,
    %kv_storage: tensor<{POOL_PAGES}x1x{PAGE_SIZE}x{HEAD_DIM}xf16>,
    %key_page_table: tensor<{BATCH}x{NUM_PAGES}xi64>,
    %value_page_table: tensor<{BATCH}x{NUM_PAGES}xi64>)
    -> tensor<{BATCH}x1x1x{HEAD_DIM}xf32> attributes {{
      hal.executable.target = #executable_target_cuda,
      translation_info = #translation
    }} {{
  %zero_f32 = arith.constant 0.0 : f32
  %neg_inf_f32 = arith.constant 0xFF800000 : f32
  %scale = arith.constant 8.837890e-02 : f16
  %key_empty = tensor.empty() : tensor<{BATCH}x{NUM_PAGES}x1x{PAGE_SIZE}x{HEAD_DIM}xf16>
  %value_empty = tensor.empty() : tensor<{BATCH}x{NUM_PAGES}x1x{PAGE_SIZE}x{HEAD_DIM}xf16>
  %key = iree_linalg_ext.gather dimension_map = [0]
      ins(%kv_storage, %key_page_table :
        tensor<{POOL_PAGES}x1x{PAGE_SIZE}x{HEAD_DIM}xf16>,
        tensor<{BATCH}x{NUM_PAGES}xi64>)
      outs(%key_empty : tensor<{BATCH}x{NUM_PAGES}x1x{PAGE_SIZE}x{HEAD_DIM}xf16>)
      -> tensor<{BATCH}x{NUM_PAGES}x1x{PAGE_SIZE}x{HEAD_DIM}xf16>
  %value = iree_linalg_ext.gather dimension_map = [0]
      ins(%kv_storage, %value_page_table :
        tensor<{POOL_PAGES}x1x{PAGE_SIZE}x{HEAD_DIM}xf16>,
        tensor<{BATCH}x{NUM_PAGES}xi64>)
      outs(%value_empty : tensor<{BATCH}x{NUM_PAGES}x1x{PAGE_SIZE}x{HEAD_DIM}xf16>)
      -> tensor<{BATCH}x{NUM_PAGES}x1x{PAGE_SIZE}x{HEAD_DIM}xf16>
  %output_empty = tensor.empty() : tensor<{BATCH}x1x1x{HEAD_DIM}xf32>
  %max_empty = tensor.empty() : tensor<{BATCH}x1x1xf32>
  %sum_empty = tensor.empty() : tensor<{BATCH}x1x1xf32>
  %output_init = linalg.fill ins(%zero_f32 : f32)
      outs(%output_empty : tensor<{BATCH}x1x1x{HEAD_DIM}xf32>)
      -> tensor<{BATCH}x1x1x{HEAD_DIM}xf32>
  %max_init = linalg.fill ins(%neg_inf_f32 : f32)
      outs(%max_empty : tensor<{BATCH}x1x1xf32>) -> tensor<{BATCH}x1x1xf32>
  %sum_init = linalg.fill ins(%zero_f32 : f32)
      outs(%sum_empty : tensor<{BATCH}x1x1xf32>) -> tensor<{BATCH}x1x1xf32>
  %attention:3 = iree_linalg_ext.online_attention {{
      decomposition_config = {{
        pv_attrs = {{lowering_config = #pv_attrs_config}},
        qk_attrs = {{lowering_config = #qk_attrs_config}}
      }},
      indexing_maps = [
        affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1, d2, d4)>,
        affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d5, d1, d6, d4)>,
        affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d5, d1, d6, d3)>,
        affine_map<(d0, d1, d2, d3, d4, d5, d6) -> ()>,
        affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1, d2, d3)>,
        affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1, d2)>,
        affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1, d2)>],
      lowering_config = #attention_lowering_config
    }}
    ins(%query, %key, %value, %scale :
      tensor<{BATCH}x1x1x{HEAD_DIM}xf16>,
      tensor<{BATCH}x{NUM_PAGES}x1x{PAGE_SIZE}x{HEAD_DIM}xf16>,
      tensor<{BATCH}x{NUM_PAGES}x1x{PAGE_SIZE}x{HEAD_DIM}xf16>, f16)
    outs(%output_init, %max_init, %sum_init :
      tensor<{BATCH}x1x1x{HEAD_DIM}xf32>, tensor<{BATCH}x1x1xf32>,
      tensor<{BATCH}x1x1xf32>) {{
      ^bb0(%score: f32):
        iree_linalg_ext.yield %score : f32
    }} -> tensor<{BATCH}x1x1x{HEAD_DIM}xf32>, tensor<{BATCH}x1x1xf32>, tensor<{BATCH}x1x1xf32>
  %normalized_empty = tensor.empty() : tensor<{BATCH}x1x1x{HEAD_DIM}xf32>
  %normalized = linalg.generic {{
      indexing_maps = [
        affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
        affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>,
        affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
      iterator_types = ["parallel", "parallel", "parallel", "parallel"]
    }} ins(%attention#0, %attention#2 :
      tensor<{BATCH}x1x1x{HEAD_DIM}xf32>, tensor<{BATCH}x1x1xf32>)
    outs(%normalized_empty : tensor<{BATCH}x1x1x{HEAD_DIM}xf32>) {{
      ^bb0(%attn_value: f32, %sum: f32, %out: f32):
        %result = arith.divf %attn_value, %sum : f32
        linalg.yield %result : f32
    }} -> tensor<{BATCH}x1x1x{HEAD_DIM}xf32>
  return %normalized : tensor<{BATCH}x1x1x{HEAD_DIM}xf32>
}}
"""
    with open(output_path, "w") as file:
        file.write(text)


def generate_calls_mlir(output_path):
    key_table = [3, 0, 6, 1, 4, 7, 2, 5, 1, 6, 3, 0, 7, 2, 5, 4]
    value_table = [5, 2, 7, 0, 6, 1, 4, 3, 2, 7, 0, 5, 3, 6, 1, 4]
    key_values = ", ".join(str(value) for value in key_table)
    value_values = ", ".join(str(value) for value in value_table)
    text = f"""\
builtin.module @calls {{
  func.func private @attention_test.generate_random_4d_tensor(
      %device: !hal.device, %dim0: i64, %dim1: i64, %dim2: i64,
      %dim3: i64, %element_type: i32, %seed: i32) -> !hal.buffer_view
  func.func private @attention_test.check_paged_attention_results(
      %device: !hal.device, %batch: i64, %num_pages: i64, %page_size: i64,
      %head_dim: i64, %query: !hal.buffer_view, %kv_storage: !hal.buffer_view,
      %key_page_table: !hal.buffer_view, %value_page_table: !hal.buffer_view,
      %result: !hal.buffer_view)
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
    %c1 = arith.constant 1 : i64
    %c8 = arith.constant {POOL_PAGES} : i64
    %c17 = arith.constant 17 : i32
    %c23 = arith.constant 23 : i32
    %query_type = hal.element_type<f16> : i32
    %query = call @attention_test.generate_random_4d_tensor(
      %device, %batch, %c1, %c1, %head_dim, %query_type, %c17) : (!hal.device, i64, i64, i64, i64, i32, i32) -> !hal.buffer_view
    %kv = call @attention_test.generate_random_4d_tensor(
      %device, %c8, %c1, %page_size, %head_dim, %query_type, %c23) : (!hal.device, i64, i64, i64, i64, i32, i32) -> !hal.buffer_view
    %key_flat = arith.constant dense<[{key_values}]> : tensor<{BATCH * NUM_PAGES}xi64>
    %key_table = tensor.expand_shape %key_flat [[0, 1]] output_shape [{BATCH}, {NUM_PAGES}] : tensor<{BATCH * NUM_PAGES}xi64> into tensor<{BATCH}x{NUM_PAGES}xi64>
    %key_page_table = hal.tensor.export %key_table : tensor<{BATCH}x{NUM_PAGES}xi64> -> !hal.buffer_view
    %value_flat = arith.constant dense<[{value_values}]> : tensor<{BATCH * NUM_PAGES}xi64>
    %value_table = tensor.expand_shape %value_flat [[0, 1]] output_shape [{BATCH}, {NUM_PAGES}] : tensor<{BATCH * NUM_PAGES}xi64> into tensor<{BATCH}x{NUM_PAGES}xi64>
    %value_page_table = hal.tensor.export %value_table : tensor<{BATCH}x{NUM_PAGES}xi64> -> !hal.buffer_view
    %result = call @module.paged_attention(%query, %kv, %key_page_table, %value_page_table) : (!hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view) -> !hal.buffer_view
    call @attention_test.check_paged_attention_results(%device, %batch, %num_pages, %page_size, %head_dim, %query, %kv, %key_page_table, %value_page_table, %result) : (!hal.device, i64, i64, i64, i64, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view) -> ()
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

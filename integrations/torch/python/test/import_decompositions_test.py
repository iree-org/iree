# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import sys

import pytest
from iree.importers.torch.import_decompositions import import_decompositions


def test_import_decompositions():
    expected_decompositions = [
        "aten.embedding_dense_backward.default",
        "aten.embedding_dense_backward.out",
        "aten.native_layer_norm_backward.default",
        "aten.native_layer_norm_backward.out",
        "aten.slice_backward.default",
        "aten.slice_backward.out",
        "aten.select_backward.default",
        "aten.select_backward.out",
        "aten.norm.ScalarOpt_dim",
        "aten.native_group_norm.default",
        "aten.upsample_bilinear2d.vec",
        "aten.split.Tensor",
        "aten.native_layer_norm.default",
        "aten.native_layer_norm.out",
        "aten.masked_fill.Tensor",
        "aten.masked_fill.Scalar",
        "aten.t.default",
        "aten.addmm.default",
        "aten.addmm.out",
        "aten.addmm.dtype_out",
        "aten.addmm.dtype",
        "aten.__and__.Tensor",
        "aten._native_batch_norm_legit_functional.default",
        "aten._native_batch_norm_legit_no_training.default",
        "aten._native_batch_norm_legit.default",
        "aten._native_batch_norm_legit.no_stats",
        "aten.squeeze.dims",
        "aten.soft_margin_loss.default",
        "aten.soft_margin_loss.out",
        "aten.im2col.default",
        "aten.im2col.out",
        "aten._euclidean_dist.default",
        "aten._euclidean_dist.out",
        "aten.index_copy.default",
        "aten.index_copy.out",
        "aten.index_copy_.default",
        "aten.grid_sampler_2d.default",
        "aten.grid_sampler_2d.out",
        "aten.log_sigmoid_forward.default",
        "aten.log_sigmoid_forward.output",
        "aten.unsafe_split.Tensor",
        "aten.binary_cross_entropy.default",
        "aten.binary_cross_entropy.out",
        "aten.dot.default",
        "aten.dot.out",
        "aten._adaptive_avg_pool2d.default",
        "aten._adaptive_avg_pool2d.out",
        "aten._prelu_kernel.default",
        "aten.full.default",
        "aten.full.out",
        "aten._log_softmax.default",
        "aten._log_softmax.out",
        "aten.nll_loss_forward.default",
        "aten.nll_loss_forward.output",
        "aten.nll_loss_backward.default",
        "aten.nll_loss_backward.grad_input",
        "aten._to_copy.default",
        "aten._to_copy.out",
        "aten._log_softmax_backward_data.default",
        "aten._log_softmax_backward_data.out",
        "aten._unsafe_index.Tensor",
        "aten.unbind.int",
        "aten.linspace.Tensor_Tensor",
        "aten._scaled_dot_product_flash_attention_for_cpu.default",
        "aten._safe_softmax.default",
        "aten.upsample_bilinear2d.out",
        "aten.upsample_bicubic2d.vec",
        "aten.upsample_bicubic2d.default",
        "aten.upsample_bicubic2d.out",
        "aten.upsample_bilinear2d.default",
        "aten.upsample_nearest2d.vec",
        "aten.upsample_nearest2d.out",
        "aten.upsample_nearest2d.default",
        "aten._weight_norm_interface.default",
        "aten._weight_norm_interface.out",
        "aten._weight_norm.default",
        "aten._scaled_dot_product_attention_math.default",
        "aten.scaled_dot_product_attention.default",
        "aten.concat.default",
        "aten.multiply.Tensor",
        "aten.erfinv.default",
        "aten.norm_except_dim.default",
        "aten.diff.default",
        "aten.linear.default",
        "aten.matmul.default",
        "aten.gru.data",
        "aten.gru.input",
        "aten.lstm.input",
        "aten.lstm.data",
        "aten.hardtanh.out",
        "aten.conv2d.default",
        "aten.conv1d.default",
        "aten.hardtanh.default",
        "aten.conv3d.default",
    ]
    with import_decompositions() as table:
        current_decompositions = [str(x) for x in table.keys()]
        assert set(expected_decompositions) == set(current_decompositions)


if __name__ == "__main__":
    args = sys.argv[1:]
    ret_code = pytest.main([__file__, "-vv"] + args)

    sys.exit(ret_code)

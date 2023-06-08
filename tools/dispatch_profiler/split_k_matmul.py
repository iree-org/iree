# Copyright 2023 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from library import *
from dispatch import *
from matmul import MatmulOperation, MatmulCompilationInfo, CudaMatmulGenerator


class CudaSplitKMatmulGenerator(CudaMatmulGenerator):
  """SplitK Matmul dispatch generator class."""

  def __init__(self, args):
    """Initializes the splitK matmul generator."""
    super().__init__(args)

    # Predefined matmul shapes for splitK matmul.
    self.matmul_shapes = [[128, 128, 12288]]

    # Predefined split_k_slices list for splitK matmul.
    self.split_k_slices = [2, 4, 16, 18]

    # SplitK matmul dispatches collection list.
    self.dispatches_collection_list = []

  def _append_matmul_dispatch_collection(self, matmul_shapes, split_k_slices,
                                         data_type, configuration_list):
    """Appends the split-k matmul dispatches collection with the given configuration list."""

    # Create dispatches collection for each matmul_shape x split_k_slice x configuration list.
    for matmul_shape in matmul_shapes:
      for split_k_slice in split_k_slices:
        operation = MatmulOperation(
          matmul_shape,\
          TensorDescription(data_type[0], LayoutType.RowMajor), \
          TensorDescription(data_type[1], LayoutType.RowMajor), \
          TensorDescription(data_type[2], LayoutType.RowMajor), \
          1, # batch_count 
          split_k_slice,
          OperationKind.SplitkMatmul)

        # Filter out configurations that are not supported by LLVM GPU CUDA backend.
        supported_configuration_list = self._cuda_supported_configuration_list(
            operation, configuration_list)

        # Add default configuration if enabled.
        if self.args.default_config:
          supported_configuration_list.append(
              MatmulCompilationInfo([], [], OperationKind.Matmul,
                                    CompilationConfigType.Default))

        # Append the dispatch collection.
        self.dispatches_collection_list.append(DispatchCollection(\
          operation, supported_configuration_list))

  def _cuda_matmul_tensor_cores_f16(self):
    """Appends a list of matmul split-k dispatches for GPU TensorCore F16 data type."""
    configuration_list = self._get_matmul_custom_compilation_info_list(
        self.tile_descriptions_tensor_cores_f16, self.translation_infos,
        OperationKind.SplitkMatmul)
    data_type = [DataType.f16, DataType.f16, DataType.f16]
    self._append_matmul_dispatch_collection(self.matmul_shapes,
                                            self.split_k_slices, data_type,
                                            configuration_list)

  def _cuda_matmul_tensor_cores_f32(self):
    """Appends a list of matmul split-k dispatches for GPU TensorCore F32 data type."""
    configuration_list = self._get_matmul_custom_compilation_info_list(
        self.tile_descriptions_tensor_cores_f32, self.translation_infos,
        OperationKind.SplitkMatmul)
    data_type = [DataType.f32, DataType.f32, DataType.f32]
    self._append_matmul_dispatch_collection(self.matmul_shapes,
                                            self.split_k_slices, data_type,
                                            configuration_list)

  def generate(self):
    """Generates a list of split-k matmul operations."""
    self._cuda_matmul_tensor_cores_f16()
    self._cuda_matmul_tensor_cores_f32()
    return self.dispatches_collection_list

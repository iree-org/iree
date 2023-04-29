from library import *
from dispatch import *
from matmul import MatmulOperation, MatmulCompilationInfo, CudaMatmulGenerator


class CudaSplitKMatmulGenerator(CudaMatmulGenerator):
  """SplitK Matmul dispatch generator class."""

  def __init__(self, args):
    """Initializes the splitK matmul generator."""
    super().__init__(args)

    self.matmul_shapes = [[512, 1024, 4096]]
    self.split_k_slice_list = [2, 16, 18]
    self.dispatches_collection_list = []

  def _cuda_matmul_tensor_cores_f16(self):
    """Appends a list of matmul dispatches for GPU TensorCore F16 data type."""

    # Create configuration list from the tile descriptions and translation infos.
    configuration_list = []

    for tile_description in self.tile_descriptions_tensor_cores_f16:
      for translation_info in self.translation_infos:
        configuration_list.append(
            MatmulCompilationInfo(tile_description, translation_info,
                                  OperationKind.SplitkMatmul,
                                  CompilationConfigType.Custom))

    # Create dispatches collection for each problem shape with the configuration list.
    for matmul_shape in self.matmul_shapes:
      for split_k_slice in self.split_k_slice_list:
        operation = MatmulOperation(
          matmul_shape,\
          TensorDescription(DataType.f16, LayoutType.RowMajor), \
          TensorDescription(DataType.f16, LayoutType.RowMajor), \
          TensorDescription(DataType.f16, LayoutType.RowMajor), \
          1, # batch_count 
          split_k_slice,
          OperationKind.SplitkMatmul)

        # Filter out configurations that are not supported by LLVM GPU CUDA backend.
        supported_configuration_list = self._cuda_supported_configuration_list(
            operation, configuration_list)

        # Add default configuration if enabled.
        if self.args.default_config:
          supported_configuration_list.append(
              MatmulCompilationInfo([], [], OperationKind.SplitkMatmul,
                                    CompilationConfigType.Default))

        self.dispatches_collection_list.append(DispatchCollection(\
          operation, supported_configuration_list))

  def _cuda_matmul_tensor_cores_f32(self):
    """Appends a list of matmul dispatches for GPU TensorCore F32 data type."""

    # Create configuration list from the tile descriptions and translation infos.
    configuration_list = []

    for tile_description in self.tile_descriptions_tensor_cores_f32:
      for translation_info in self.translation_infos:
        configuration_list.append(
            MatmulCompilationInfo(tile_description, translation_info,
                                  OperationKind.SplitkMatmul,
                                  CompilationConfigType.Custom))

    # Create dispatches collection for each problem shape with the configuration list.
    for matmul_shape in self.matmul_shapes:
      for split_k_slice in self.split_k_slice_list:
        operation = MatmulOperation(
          matmul_shape,\
          TensorDescription(DataType.f32, LayoutType.RowMajor), \
          TensorDescription(DataType.f32, LayoutType.RowMajor), \
          TensorDescription(DataType.f32, LayoutType.RowMajor), \
          1, # batch_count 
          split_k_slice,
          OperationKind.SplitkMatmul)

        # Filter out configurations that are not supported by LLVM GPU CUDA backend.
        supported_configuration_list = self._cuda_supported_configuration_list(
            operation, configuration_list)

        # Add default configuration if enabled.
        if self.args.default_config:
          supported_configuration_list.append(
              MatmulCompilationInfo([], [], OperationKind.SplitkMatmul,
                                    CompilationConfigType.Default))

        self.dispatches_collection_list.append(DispatchCollection(\
          operation, supported_configuration_list))

  def generate(self):
    """Generates a list of matmul operations."""
    #self._cuda_matmul_tensor_cores_f16()
    self._cuda_matmul_tensor_cores_f32()
    return self.dispatches_collection_list

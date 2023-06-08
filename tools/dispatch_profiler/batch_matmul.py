# Copyright 2022 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
from library import *
from dispatch import *
from matmul import MatmulOperation, MatmulCompilationInfo, CudaMatmulGenerator


class BatchMatmulOperation(MatmulOperation):
  """Data structure to describe a batch matrix multiplication operation."""

  def __init__(self, bmm_shape, lhs, rhs, result):
    assert len(bmm_shape) == 4, "Batch matmul shape must be 4D"
    super().__init__(bmm_shape[1:], lhs, rhs, result, bmm_shape[0], 1,
                     OperationKind.BatchMatmul)

  def name(self):
    return f'{OperationKindNames[self.operation_kind]}_'\
           f'{self.batch_count}x{self.M}x{self.N}x{self.K}_'\
           f'{DataTypeName[self.lhs.datatype]}{ShortLayoutTypeName[self.lhs.layout]}_'\
           f'{DataTypeName[self.rhs.datatype]}{ShortLayoutTypeName[self.rhs.layout]}_'\
           f'{DataTypeName[self.result.datatype]}{ShortLayoutTypeName[self.result.layout]}'

  def lhs_npy_shape(self):
    return f'{self.batch_count}x{super().lhs_npy_shape()}'

  def rhs_npy_shape(self):
    return f'{self.batch_count}x{super().rhs_npy_shape()}'

  def result_npy_shape(self):
    return f'{self.batch_count}x{super().result_npy_shape()}'


class EmitLinalgBatchMatmulDispatch:
  """Emitters for the `linalg.batch_matmul` dispatch."""

  def __init__(self):
    self.mlir_dialect = MlirDialect.Linalg

    self.linalg_row_row_matmul_template = """
// Dispatch linalg.batch_matmul row-row layout 
func.func @${operation_name}_${compilation_info_name}(
  %lhs: tensor<${batch_count}x${problem_m}x${problem_k}x${datatype_lhs}>,
  %rhs: tensor<${batch_count}x${problem_k}x${problem_n}x${datatype_rhs}>) -> tensor<${batch_count}x${problem_m}x${problem_n}x${datatype_result}>
{
  %c0 = arith.constant 0.0 : ${datatype_result}
  %init = tensor.empty() : tensor<${batch_count}x${problem_m}x${problem_n}x${datatype_result}>
  %inital_result = linalg.fill ins(%c0 : ${datatype_result}) outs(%init : tensor<${batch_count}x${problem_m}x${problem_n}x${datatype_result}>) -> tensor<${batch_count}x${problem_m}x${problem_n}x${datatype_result}>
  %result = linalg.batch_matmul ${compilation_info_attribute} 
                     ins(%lhs, %rhs: tensor<${batch_count}x${problem_m}x${problem_k}x${datatype_lhs}>, tensor<${batch_count}x${problem_k}x${problem_n}x${datatype_rhs}>)
                     outs(%inital_result: tensor<${batch_count}x${problem_m}x${problem_n}x${datatype_result}>) -> tensor<${batch_count}x${problem_m}x${problem_n}x${datatype_result}>
  return %result : tensor<${batch_count}x${problem_m}x${problem_n}x${datatype_result}>
}
"""

  def emit(self, dispatch):
    """Emit the matmul operation in the MLIR dialect for a single compilation info"""
    compilation_info_attribute_template = """{compilation_info = #${compilation_info_name}}"""
    compilation_info_attribute_str = SubstituteTemplate(
        compilation_info_attribute_template,
        {'compilation_info_name': dispatch.configuration.name()})
    compilation_info_attribute = compilation_info_attribute_str \
      if dispatch.configuration.config_type != CompilationConfigType.Default else ""

    values = {
        'operation_name': dispatch.operation.name(),
        'compilation_info_attribute': compilation_info_attribute,
        'batch_count': str(dispatch.operation.batch_count),
        'problem_m': str(dispatch.operation.M),
        'problem_n': str(dispatch.operation.N),
        'problem_k': str(dispatch.operation.K),
        'datatype_lhs': DataTypeName[dispatch.operation.lhs.datatype],
        'datatype_rhs': DataTypeName[dispatch.operation.rhs.datatype],
        'datatype_result': DataTypeName[dispatch.operation.result.datatype],
        'compilation_info_name': dispatch.configuration.name()
    }

    return SubstituteTemplate(self.linalg_row_row_matmul_template, values)


class ReferenceBatchMatmulOp(ReferenceOpInterface):
  """Reference implementation for the batch matmul operation in numpy."""

  def __init__(self, bmm_operation, op_reference_cache_path, dist_lhs,
               dist_rhs):
    self.bmm_operation = bmm_operation
    self.op_reference_cache_path = op_reference_cache_path

    if not self.op_reference_cache_path.exists():
      self.op_reference_cache_path.mkdir()

    # Problem shape.
    self.batch_count = bmm_operation.batch_count
    self.M = bmm_operation.M
    self.N = bmm_operation.N
    self.K = bmm_operation.K

    # Data type for the input and result matrices.
    self.dtype_lhs = DataTypeNumPyTag[bmm_operation.lhs.datatype]
    self.dtype_rhs = DataTypeNumPyTag[bmm_operation.rhs.datatype]
    self.dtype_result = DataTypeNumPyTag[bmm_operation.result.datatype]

    # Distribution of the input tensors.
    self.dist_lhs = dist_lhs
    self.dist_rhs = dist_rhs

    # Filename for the left hand side input tensor.
    self.filename_lhs = "batch_count{batch_count}xm{problem_m}xk{problem_k}_"\
      "{tensor_description}_{dist}_lhs.npy".format(
      batch_count=self.batch_count,
      problem_m=self.M,
      problem_k=self.K,
      tensor_description=self.bmm_operation.lhs.name(),
      dist=DistributionName[self.dist_lhs])

    # Filename for the right hand side input tensor.
    self.filename_rhs = "batch_count{batch_count}xk{problem_k}xn{problem_n}_"\
      "{tensor_description}_{dist}_rhs.npy".format(
      batch_count=self.batch_count,
      problem_k=self.K,
      problem_n=self.N,
      tensor_description=self.bmm_operation.rhs.name(),
      dist=DistributionName[self.dist_rhs])

    # Filename for the reference result tensor.
    self.filename_reference_result = "batch_count{batch_count}xm{problem_m}xn{problem_n}_"\
      "{tensor_description}_reference_result.npy".format(
      batch_count=self.batch_count,
      problem_m=self.M,
      problem_n=self.N,
      tensor_description=self.bmm_operation.result.name())

    # Filepath for input and output files.
    self.filepath_lhs = self.op_reference_cache_path.joinpath(self.filename_lhs)
    self.filepath_rhs = self.op_reference_cache_path.joinpath(self.filename_rhs)
    self.filepath_reference_result = self.op_reference_cache_path.joinpath(
        self.filename_reference_result)

  def get_input_filepaths(self):
    """Returns the list of input file paths."""
    return [self.filepath_lhs, self.filepath_rhs]

  def get_output_filepaths(self):
    """Returns the list of expected output file paths."""
    return [self.filepath_reference_result]

  def __call__(self):
    """Generates input data, runs reference numpy.matmul, and save npy files to the output directory."""
    # Generate the input data as np.array for the matmul operation.
    lhs_np_array = get_np_array(self.bmm_operation.lhs,
                                (self.batch_count, self.M, self.K),
                                self.dist_lhs)
    rhs_np_array = get_np_array(self.bmm_operation.rhs,
                                (self.batch_count, self.K, self.N),
                                self.dist_rhs)

    # Run the reference np.matmul and generate result np.array.
    result = np.matmul(lhs_np_array, rhs_np_array)

    # Save the input data as np.array for the matmul operation.
    np.save(self.filepath_lhs, np.array(lhs_np_array, dtype=self.dtype_lhs))
    np.save(self.filepath_rhs, np.array(rhs_np_array, dtype=self.dtype_rhs))

    # Save the expected result as an np.array.
    np.save(self.filepath_reference_result,
            np.array(result, dtype=self.dtype_result))


##############################################################################
class CudaBatchMatmulGenerator(CudaMatmulGenerator):
  """Batch matmul dispatch generator class. """

  def __init__(self, args):
    """Initializes the batch matmul dispatch generator."""
    super().__init__(args)

    # Predefined batch matmul problem shapes.
    self.batch_matmul_shapes = [[16, 512, 64, 512]]

    # Batch matmul dispatches collection.
    self.dispatches_collection_list = []

  def _append_matmul_dispatch_collection(self, bmm_shapes, data_type,
                                         configuration_list):
    """Update the batch matmul dispatch collection with the given configuration list."""

    # Create dispatches collection for each problem shape with the configuration list.
    for bmm_shape in bmm_shapes:
      operation = BatchMatmulOperation(
        bmm_shape,\
        TensorDescription(data_type[0], LayoutType.RowMajor), \
        TensorDescription(data_type[1], LayoutType.RowMajor), \
        TensorDescription(data_type[2], LayoutType.RowMajor))

      # Filter out configurations that are not supported by LLVM GPU CUDA backend.
      supported_configuration_list = self._cuda_supported_configuration_list(
          operation, configuration_list)

      # Add default configuration if enabled.
      if self.args.default_config:
        supported_configuration_list.append(
            MatmulCompilationInfo([], [], OperationKind.BatchMatmul,
                                  CompilationConfigType.Default))

      # Append the dispatches collection.
      self.dispatches_collection_list.append(DispatchCollection(\
        operation, supported_configuration_list))

  def _cuda_matmul_tensor_cores_f16(self):
    """Appends a list of matmul dispatches for GPU TensorCore F16 data type."""
    configuration_list = self._get_matmul_custom_compilation_info_list(
        self.tile_descriptions_tensor_cores_f16, self.translation_infos,
        OperationKind.BatchMatmul)
    data_type = [DataType.f16, DataType.f16, DataType.f16]
    self._append_matmul_dispatch_collection(self.batch_matmul_shapes, data_type,
                                            configuration_list)

  def _cuda_matmul_tensor_cores_f32(self):
    """Appends a list of matmul dispatches for GPU TensorCore F32 data type."""
    configuration_list = self._get_matmul_custom_compilation_info_list(
        self.tile_descriptions_tensor_cores_f32, self.translation_infos,
        OperationKind.BatchMatmul)
    data_type = [DataType.f32, DataType.f32, DataType.f32]
    self._append_matmul_dispatch_collection(self.batch_matmul_shapes, data_type,
                                            configuration_list)

  def generate(self):
    """Generates a list of matmul operations."""
    self._cuda_matmul_tensor_cores_f16()
    self._cuda_matmul_tensor_cores_f32()
    return self.dispatches_collection_list

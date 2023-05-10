# Copyright 2023 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import enum, shutil, functools, operator, collections, subprocess
from library import *
from dispatch import *
from options import get_cmd_line_argument_list


################################################################################
class MatmulOperation:
  """Data structure to describe a matrix multiplication operation. 
     This includes the shape, datatype, and layout of the operands. This data 
     structure is *independent* of the compilation* and tiling configuration. 
     It "mostly" contains the parameter that changes the functionality of matmul 
     operation. The only exception is the split_k_slices parameter, which is
     changes the performance of the matmul operation and not the functionality.
  """

  def __init__(self,
               matmul_shape,
               lhs,
               rhs,
               result,
               batch_count=1,
               split_k_slices=1,
               operation_kind=OperationKind.Matmul):
    """Initializes a matrix multiplication operation.
    Matrix-multiple operation: `result[M, N] = lhs[M, K] * rhs[K, N]`
    matmul_shape: A tuple representing the matrix multiplication problem shape
      in the format (M, N, K), where M is the number of rows in the lhs matrix, 
      N is the number of columns in the rhs matrix, and K is the number of columns 
      in the lhs matrix and rows in the rhs matrix.
    lhs: A TensorDescription object representing the left-hand-side matrix operand.
    rhs: A TensorDescription object representing the right-hand-side matrix operand.
    result: A TensorDescription object representing the result matrix operand.
    """

    # Parameters that change the matmul operation *functionally*.
    self.operation_kind = operation_kind
    self.matmul_shape = matmul_shape
    self.M = matmul_shape[0]
    self.N = matmul_shape[1]
    self.K = matmul_shape[2]
    self.batch_count = batch_count
    self.lhs = lhs  # TensorDescription
    self.rhs = rhs  # TensorDescription
    self.result = result  # TensorDescription

    # Parameters that change the matmul operation *performance*.
    self.split_k_slices = split_k_slices

  def __eq__(self, other):
    """Returns true if the matmul operation is *functionally* the same."""
    return self.matmul_shape == other.matmul_shape and \
           self.lhs == other.lhs and \
           self.rhs == other.rhs and \
           self.result == other.result and \
           self.batch_count == other.batch_count

  def name(self):
    """Procedurally generated name for the matmul operation.
    The name uniquely identifies a matmul operation with matmul shape, 
    lhs dataype and layout, rhs datatype and layout, and result
    datatype and layout.
    """
    return f'{OperationKindNames[self.operation_kind]}_'\
           f'{self.M}x{self.N}x{self.K}_'\
           f'{DataTypeName[self.lhs.datatype]}{ShortLayoutTypeName[self.lhs.layout]}_'\
           f'{DataTypeName[self.rhs.datatype]}{ShortLayoutTypeName[self.rhs.layout]}_'\
           f'{DataTypeName[self.result.datatype]}{ShortLayoutTypeName[self.result.layout]}'

  def get_argument_dict(self):
    """Returns the dictionary of matmul arguments (shape, datatypes, split_k_slices)."""
    split_k_mode = "parallel" if self.operation_kind == OperationKind.SplitkMatmul else "N/A"
    split_k_slices = self.split_k_slices if self.operation_kind == OperationKind.SplitkMatmul else "N/A"
    return {
        "batch_count": self.batch_count,
        "m": self.M,
        "n": self.N,
        "k": self.K,
        "lhs": self.lhs.name(),
        "rhs": self.rhs.name(),
        "result": self.result.name(),
        "split_k_mode": split_k_mode,
        "split_k_slices": split_k_slices
    }

  def get_dict_entry(self):
    """Returns the dictionary of matmul operation summary."""
    dict_entry = {
        "op_kind": OperationKindNames[self.operation_kind],
        "Operation": self.name(),
        "bytes": self.bytes(),
        "flops": self.flops(),
    }
    dict_entry.update(self.get_argument_dict())
    return dict_entry

  def lhs_npy_shape(self):
    """Returns the shape of the lhs numpy array as a string in the format "MxKxDataType"."""
    return f"{self.M}x{self.K}x{DataTypeName[self.lhs.datatype]}"

  def rhs_npy_shape(self):
    """Returns the shape of the rhs numpy array as a string in the format "KxNxDataType"."""
    return f"{self.K}x{self.N}x{DataTypeName[self.rhs.datatype]}"

  def result_npy_shape(self):
    """Returns the shape of the result numpy array as a string in the format "MxNxDataType"."""
    return f"{self.M}x{self.N}x{DataTypeName[self.result.datatype]}"

  def bytes(self):
    """Returns the number of bytes read/written by the matmul operation."""
    bytes = (DataTypeSizeInBits[self.lhs.datatype] * self.M // 8) * self.K + \
            (DataTypeSizeInBits[self.rhs.datatype] * self.K // 8) * self.N + \
            (DataTypeSizeInBits[self.result.datatype] * self.M // 8) * self.N
    return bytes * self.batch_count

  def flops(self):
    """Returns the number of floating point operations performed by the matmul operation."""
    return 2 * self.M * self.N * self.K * self.batch_count


##############################################################################
class MatmulCompilationInfo:
  """Data structure strictly describes the compilation passes and the tiling configurations. 
  For a matrix multiplication operation, compilation passes and tiling configuration 
  influences the performance of the compiled matmul operation, but the functionality. 
  This data structure should be independent of the matmul operation functionality. 
  
  Any change in this data structure should not affect the functionality of the matmul operation, i.e., 
  we should be able to use the same reference results for a matrix operation compiled with different 
  compilation info.
  """

  def __init__(self,
               tile_description,
               translation_info,
               operation_kind=OperationKind.Matmul,
               config_type=CompilationConfigType.Custom):
    self.tile_description = tile_description  # TileDescription
    self.translation_info = translation_info  # TranslationInfo
    self.operation_kind = operation_kind  # OperationKind
    self.config_type = config_type  # CompilationConfigType

  def get_dict_entry(self):
    """Returns the dictionary entry for the matmul compilation info."""
    if self.config_type == CompilationConfigType.Default:
      return {
          "Tile config": "Default",
          "Core class": "Default",
          "Instruction class": "Default"
      }

    translation_info_name = TranslationInfoName[self.translation_info]
    return {
        "Tile config": self.tile_description.name(),
        "Core class": translation_info_name.split('_')[0],
        "Instruction class": translation_info_name.split('_')[1],
    }

  def name(self):
    """Procedurally generated name for the matmul compilation info."""
    if self.config_type == CompilationConfigType.Default:
      return "tile_config_default"

    return "tile_config_{tbm}x{tbn}_{tbk}x{stages}_{translation_info}".format(
        tbm=self.tile_description.threadblock_shape[0],
        tbn=self.tile_description.threadblock_shape[1],
        tbk=self.tile_description.threadblock_shape[2],
        stages=self.tile_description.stages,
        translation_info=TranslationInfoName[self.translation_info])


################################################################################
class EmitMatmulCompilationInfo:
  """Emitters for the matmul compilation info."""

  def __init__(self):
    # matmul compilation info template
    self.matmul_compilation_info_template = """
// matmul compilation info (tile configuration, translation info, workgroup size)
#${compilation_info_name} = #iree_codegen.compilation_info<
  lowering_config = <tile_sizes = [[${threadblock_shape_m}, ${threadblock_shape_n}, ${threadblock_shape_k}]]>,
  translation_info = <${translation_info} pipeline_depth = ${stages}>,
  workgroup_size = [${block_dim_x} : index, ${block_dim_y} : index, ${block_dim_z} : index]
>
"""
    # batch matmul and split-k matmul compilation info template
    self.batch_matmul_compilation_info_template = """
// batch matmul compilation info (tile configuration, translation info, workgroup size)
#${compilation_info_name} = #iree_codegen.compilation_info<
  lowering_config = <tile_sizes = [[1, ${threadblock_shape_m}, ${threadblock_shape_n}, ${threadblock_shape_k}]]>,
  translation_info = <${translation_info} pipeline_depth = ${stages}>,
  workgroup_size = [${block_dim_x} : index, ${block_dim_y} : index, ${block_dim_z} : index]
>
"""

  def emit(self, compilation_info):
    """Emits the matmul compilation info as a string."""
    if compilation_info.config_type == CompilationConfigType.Default:
      return ""

    values = {
        'compilation_info_name':
            compilation_info.name(),
        'translation_info':
            TranslationInfoTag[compilation_info.translation_info],
        'threadblock_shape_m':
            str(compilation_info.tile_description.threadblock_shape[0]),
        'threadblock_shape_n':
            str(compilation_info.tile_description.threadblock_shape[1]),
        'threadblock_shape_k':
            str(compilation_info.tile_description.threadblock_shape[2]),
        'stages':
            str(compilation_info.tile_description.stages),
        'block_dim_x':
            str(compilation_info.tile_description.block_dim[0]),
        'block_dim_y':
            str(compilation_info.tile_description.block_dim[1]),
        'block_dim_z':
            str(compilation_info.tile_description.block_dim[2]),
    }

    # linalg.matmul (without split-k) compilation info template.
    compilation_info_template = self.matmul_compilation_info_template

    # linalg.batch_matmul and linalg.matmul (with split-k) have different
    # compilation info template from the linalg.matmul (without split-k).
    if compilation_info.operation_kind == OperationKind.BatchMatmul or \
       compilation_info.operation_kind == OperationKind.SplitkMatmul:
      compilation_info_template = self.batch_matmul_compilation_info_template

    return SubstituteTemplate(compilation_info_template, values)


###############################################################################
class EmitLinalgMatmulDispatch:
  """Emitters for the `linalg.matmul` dispatch."""

  def __init__(self):
    self.mlir_dialect = MlirDialect.Linalg

    # linalg.matmul mlir template
    self.linalg_row_row_matmul_template = """
// Dispatch linalg.matmul row-row layout 
func.func @${operation_name}_${compilation_info_name}(
  %lhs: tensor<${problem_m}x${problem_k}x${datatype_lhs}>,
  %rhs: tensor<${problem_k}x${problem_n}x${datatype_rhs}>) -> tensor<${problem_m}x${problem_n}x${datatype_result}>
{
  %c0 = arith.constant 0.0 : ${datatype_result}
  %init = tensor.empty() : tensor<${problem_m}x${problem_n}x${datatype_result}>
  %inital_result = linalg.fill ins(%c0 : ${datatype_result}) outs(%init : tensor<${problem_m}x${problem_n}x${datatype_result}>) -> tensor<${problem_m}x${problem_n}x${datatype_result}>
  %result = linalg.matmul ${compilation_info_attribute} 
                     ins(%lhs, %rhs: tensor<${problem_m}x${problem_k}x${datatype_lhs}>, tensor<${problem_k}x${problem_n}x${datatype_rhs}>)
                     outs(%inital_result: tensor<${problem_m}x${problem_n}x${datatype_result}>) -> tensor<${problem_m}x${problem_n}x${datatype_result}>
  return %result : tensor<${problem_m}x${problem_n}x${datatype_result}>
}
"""

  def emit(self, matmul_dispatch):
    """Emit the matmul operation in the MLIR dialect for a single compilation info"""
    compilation_info_attribute_template = """{compilation_info = #${compilation_info_name}}"""
    compilation_info_attribute_str = SubstituteTemplate(
        compilation_info_attribute_template,
        {'compilation_info_name': matmul_dispatch.configuration.name()})
    compilation_info_attribute = compilation_info_attribute_str \
      if matmul_dispatch.configuration.config_type != CompilationConfigType.Default else ""

    values = {
        'operation_name':
            matmul_dispatch.operation.name(),
        'compilation_info_attribute':
            compilation_info_attribute,
        'problem_m':
            str(matmul_dispatch.operation.M),
        'problem_n':
            str(matmul_dispatch.operation.N),
        'problem_k':
            str(matmul_dispatch.operation.K),
        'datatype_lhs':
            DataTypeName[matmul_dispatch.operation.lhs.datatype],
        'datatype_rhs':
            DataTypeName[matmul_dispatch.operation.rhs.datatype],
        'datatype_result':
            DataTypeName[matmul_dispatch.operation.result.datatype],
        'compilation_info_name':
            matmul_dispatch.configuration.name()
    }

    return SubstituteTemplate(self.linalg_row_row_matmul_template, values)


###############################################################################
class ReferenceMatmulOp(ReferenceOpInterface):
  """Reference implementation for the matmul operation in numpy."""

  def __init__(self, matmul_operation, op_reference_cache_path, dist_lhs,
               dist_rhs):
    self.matmul_operation = matmul_operation
    self.op_reference_cache_path = op_reference_cache_path

    # Problem shape.
    self.M = matmul_operation.M
    self.N = matmul_operation.N
    self.K = matmul_operation.K

    # Data type for the input and result matrices.
    self.dtype_lhs = DataTypeNumPyTag[matmul_operation.lhs.datatype]
    self.dtype_rhs = DataTypeNumPyTag[matmul_operation.rhs.datatype]
    self.dtype_result = DataTypeNumPyTag[matmul_operation.result.datatype]

    # Distribution of the input tensors.
    self.dist_lhs = dist_lhs
    self.dist_rhs = dist_rhs

    # Filename for the left hand side input tensor.
    self.filename_lhs = "m{problem_m}xk{problem_k}_"\
      "{tensor_description}_{dist}_lhs.npy".format(
      problem_m=self.M,
      problem_k=self.K,
      tensor_description=self.matmul_operation.lhs.name(),
      dist=DistributionName[self.dist_lhs])

    # Filename for the right hand side input tensor.
    self.filename_rhs = "k{problem_k}xn{problem_n}_"\
      "{tensor_description}_{dist}_rhs.npy".format(
      problem_k=self.K,
      problem_n=self.N,
      tensor_description=self.matmul_operation.rhs.name(),
      dist=DistributionName[self.dist_rhs])

    # Filename for the reference result tensor.
    self.filename_reference_result = "m{problem_m}xn{problem_n}_"\
      "{tensor_description}_reference_result.npy".format(
      problem_m=self.M,
      problem_n=self.N,
      tensor_description=self.matmul_operation.result.name())

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
    lhs_np_array = get_np_array(self.matmul_operation.lhs, (self.M, self.K),
                                self.dist_lhs)
    rhs_np_array = get_np_array(self.matmul_operation.rhs, (self.K, self.N),
                                self.dist_rhs)

    # Run the reference np.matmul and generate result np.array.
    result = np.matmul(lhs_np_array, rhs_np_array)

    # Save the input data as np.array for the matmul operation.
    np.save(self.filepath_lhs, np.array(lhs_np_array, dtype=self.dtype_lhs))
    np.save(self.filepath_rhs, np.array(rhs_np_array, dtype=self.dtype_rhs))

    # Save the expected result as an np.array.
    np.save(self.filepath_reference_result,
            np.array(result, dtype=self.dtype_result))


class CudaMatmulDispatchChecker:
  """Given a matmul dispatch, checks if the dispatch is supported by the target GPU."""

  def __init__(self, args):
    self.args = args

    # CUDA shared memory capacity per SM in KB.
    self.sharedMemPerSm = {
        "sm_80": 163,  # 1KB is reserved for the driver.
        "sm_86": 99,  # 1KB is reserved for the driver
    }

    self.cuda_arch = self.args.cuda_arch
    self.cuda_smem_capacity_in_bytes = self.sharedMemPerSm[self.cuda_arch] << 10

  def _is_tile_aligned_shape(self, dispatch):
    """Checks if the given dispatch is valid for CUDA."""
    matmul_shape = dispatch.operation.matmul_shape
    threadblock_shape = dispatch.configuration.tile_description.threadblock_shape
    if len(matmul_shape) != len(threadblock_shape):
      raise ValueError(
          "Problem shape and threadblock shape must have the same rank.")
    is_aligned = all(
        a % b == 0 for a, b in zip(matmul_shape, threadblock_shape))
    return is_aligned

  def _cuda_smem_required_in_bytes(self, dispatch):
    """Returns size bytes of shared memory required for a given cuda dispatch."""
    threadblock_shape = dispatch.configuration.tile_description.threadblock_shape
    num_stages = dispatch.configuration.tile_description.stages
    tile_shape_lhs = threadblock_shape[0] * threadblock_shape[2]
    tile_shape_rhs = threadblock_shape[2] * threadblock_shape[1]
    return (
        (tile_shape_lhs * DataTypeSizeInBits[dispatch.operation.lhs.datatype] +
         tile_shape_rhs * DataTypeSizeInBits[dispatch.operation.rhs.datatype]) *
        num_stages) // 8

  def _is_problem_k_divisible_by_split_k(self, dispatch):
    """Checks if the given dispatch is valid for CUDA."""
    return dispatch.operation.K % dispatch.operation.split_k_slices == 0

  def _is_cuda_smem_avialable(self, dispatch):
    """Checks if the given dispatch is valid for CUDA."""
    return self._cuda_smem_required_in_bytes(
        dispatch) <= self.cuda_smem_capacity_in_bytes

  def is_valid(self, dispatch):
    """Checks if the given dispatch is valid for CUDA."""
    if not self._is_tile_aligned_shape(dispatch):
      if self.args.verbose:
        print(f"[Warning]: {dispatch.name()} is not aligned is being skipped.")
      return False
    if not self._is_cuda_smem_avialable(dispatch):
      if self.args.verbose:
        print(f"[Warning]: {dispatch.name()} requires {self._cuda_smem_required_in_bytes(dispatch)} "\
              f"bytes of shared memory, which is larger than the {self.cuda_arch} capacity "\
              f"{self.cuda_smem_capacity_in_bytes} bytes.")
      return False
    if (dispatch.operation.split_k_slices >
        1) and (not self._is_problem_k_divisible_by_split_k(dispatch)):
      if self.args.verbose:
        print(f"[Warning]: {dispatch.name()} problem k is not divisible by {dispatch.operation.split_k_slices} "\
              f"split-k slices, which is not supported on LLVM GPU CUDA backend.")
      return False
    return True


class CudaMatmulGenerator:
  """Matmul dispatch generator class.
  Generates a list of pre-defined matmul operations with resonable tuning cofigurations. 
  The generator function are seperated based on the target backend and the data type.
  Please see example `MatmulGenerator._cuda_matmul_tensor_cores_f16` for cuda target 
  backend and f16 data type."""

  def __init__(self, args):
    """Initializes the matmul generator."""
    self.args = args
    self.translation_infos = [
        #TranslationInfo.LLVMGPUMatmulSimt,  # CUDA Core (SMIT)
        #TranslationInfo.LLVMGPUMatmulTensorCore, # Tensor Core (WMMA)
        TranslationInfo.
        LLVMGPUMatmulTensorCoreMmaSync,  # Tensor Core (MMA.SYNC)
    ]

    # List of pre-defined threadblock tile shapes for Tensor Core.
    self.tile_descriptions_tensor_cores_f16 = [
        TileDescription([256, 128, 32], 3, [64, 4, 1]),
        TileDescription([128, 256, 32], 3, [128, 2, 1]),
        TileDescription([128, 128, 64], 4, [64, 2, 1]),
        TileDescription([128, 128, 32], 5, [64, 2, 1]),
        TileDescription([128, 64, 32], 5, [64, 2, 1]),
        TileDescription([64, 64, 64], 5, [64, 2, 1]),
        TileDescription([64, 64, 32], 10, [64, 2, 1]),
    ]

    self.tile_descriptions_tensor_cores_f32 = [
        TileDescription([128, 256, 16], 3, [128, 2, 1]),
        TileDescription([256, 128, 16], 3, [64, 4, 1]),
        TileDescription([128, 128, 16], 5, [64, 2, 1]),
        TileDescription([128, 128, 32], 3, [64, 2, 1]),
        TileDescription([128, 128, 32], 4, [64, 2, 1]),
        TileDescription([128, 64, 32], 3, [64, 2, 1]),
        TileDescription([128, 64, 16], 5, [64, 2, 1]),
        TileDescription([64, 64, 32], 3, [64, 2, 1]),
        TileDescription([64, 64, 16], 10, [64, 2, 1]),
    ]

    # Create a list of matmul problem and initialize with some *default* shapes.
    self.matmul_shapes = [[128, 128, 256], [256, 512, 128], [1024, 512, 2048],
                          [2560, 2560, 2560], [3456, 1024, 2048]]

    # Append matmul problem with *user* provided shapes.
    for m in get_cmd_line_argument_list(self.args.problem_m):
      for n in get_cmd_line_argument_list(self.args.problem_n):
        for k in get_cmd_line_argument_list(self.args.problem_k):
          self.matmul_shapes.append([m, n, k])

    # Matmul dispatches collection.
    self.dispatches_collection_list = []

  def _cuda_supported_configuration_list(self, operation, configuration_list):
    """Returns a list of supported configurations for CUDA."""
    supported_configuration_list = []
    dispatch_checker = CudaMatmulDispatchChecker(self.args)
    for configuration in configuration_list:
      if not dispatch_checker.is_valid(Dispatch(operation, configuration)):
        continue
      supported_configuration_list.append(configuration)

    # Return the supported configuration list.
    return supported_configuration_list

  def _get_matmul_custom_compilation_info_list(self, tile_descriptions,
                                               translation_infos,
                                               operation_kind):
    """Creates a *custom* list of matmul compilation info."""
    configuration_list = []
    for tile_description in tile_descriptions:
      for translation_info in translation_infos:
        configuration_list.append(
            MatmulCompilationInfo(tile_description, translation_info,
                                  operation_kind, CompilationConfigType.Custom))
    return configuration_list

  def _append_matmul_dispatch_collection(self, matmul_shapes, data_type,
                                         configuration_list):
    """Appends the matmul dispatches collection with the given configuration list."""

    # Create dispatches collection for each matmul_shape x configuration list..
    for matmul_shape in matmul_shapes:
      operation = MatmulOperation(
        matmul_shape,\
        TensorDescription(data_type[0], LayoutType.RowMajor), \
        TensorDescription(data_type[1], LayoutType.RowMajor), \
        TensorDescription(data_type[2], LayoutType.RowMajor))

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
    """Appends a list of matmul dispatches for GPU TensorCore F16 data type."""
    configuration_list = self._get_matmul_custom_compilation_info_list(
        self.tile_descriptions_tensor_cores_f16, self.translation_infos,
        OperationKind.Matmul)
    data_type = [DataType.f16, DataType.f16, DataType.f16]
    self._append_matmul_dispatch_collection(self.matmul_shapes, data_type,
                                            configuration_list)

  def _cuda_matmul_tensor_cores_f32(self):
    """Appends a list of matmul dispatches for GPU TensorCore F32 data type."""
    configuration_list = self._get_matmul_custom_compilation_info_list(
        self.tile_descriptions_tensor_cores_f32, self.translation_infos,
        OperationKind.Matmul)
    data_type = [DataType.f32, DataType.f32, DataType.f32]
    self._append_matmul_dispatch_collection(self.matmul_shapes, data_type,
                                            configuration_list)

  def generate(self):
    """Generates a list of matmul operations."""
    self._cuda_matmul_tensor_cores_f16()
    self._cuda_matmul_tensor_cores_f32()
    return self.dispatches_collection_list

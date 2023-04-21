from matmul import *


class BatchMatmulOperation(MatmulOperation):

  def __init__(self, bmm_shape, lhs, rhs, result):
    super().__init__(bmm_shape[1:], lhs, rhs, result)
    self.batch_size = bmm_shape[0]
    self.operation_kind = OperationKind.BatchMatmul

  def name(self):
    return f'{OperationKindNames[self.operation_kind]}_'\
           f'{self.batch_size}x{self.M}x{self.N}x{self.K}_'\
           f'{DataTypeName[self.lhs.datatype]}{ShortLayoutTypeName[self.lhs.layout]}_'\
           f'{DataTypeName[self.rhs.datatype]}{ShortLayoutTypeName[self.rhs.layout]}_'\
           f'{DataTypeName[self.result.datatype]}{ShortLayoutTypeName[self.result.layout]}'

  def csv_headers(self):
    return ["Batch size", "M", "N", "K", "lhs", "rhs", "result"]

  def create_dict_entry(self):
    return [self.batch_size] + super().create_dict_entry()

  def lhs_npy_shape(self):
    return f'{self.batch_size}x{super().lhs_npy_shape()}'

  def rhs_npy_shape(self):
    return f'{self.batch_size}x{super().rhs_npy_shape()}'

  def result_npy_shape(self):
    return f'{self.batch_size}x{super().result_npy_shape()}'

  def bytes(self):
    return self.batch_size * super().bytes()

  def flops(self):
    return self.batch_size * super().flops()


class BatchMatmulCompilationInfo(MatmulCompilationInfo):
  """Compilation info for a batch matmul operation."""

  def __init__(self,
               tile_description,
               translation_info,
               config_type=CompilationConfigType.Custom):
    super().__init__(tile_description, translation_info, config_type)

  def csv_headers(self):
    """Returns the csv headers for the matmul compilation info."""
    return super().csv_headers()

  def create_dict_entry(self):
    """Returns the dictionary entry for the matmul compilation info."""
    return super().create_dict_entry()

  def name(self):
    """Procedurally generated name for the matmul compilation info."""
    if self.config_type == CompilationConfigType.Default:
      return "tile_config_default"

    return "tile_config_1x{tbm}x{tbn}_{tbk}x{stages}_{translation_info}".format(
        tbm=self.tile_description.threadblock_shape[0],
        tbn=self.tile_description.threadblock_shape[1],
        tbk=self.tile_description.threadblock_shape[2],
        stages=self.tile_description.stages,
        translation_info=TranslationInfoName[self.translation_info])


class EmitBatchMatmulCompilationInfo:
  """Emitters for the matmul compilation info."""

  def __init__(self):
    self.compilation_info_template = """
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

    return SubstituteTemplate(self.compilation_info_template, values)


class EmitLinalgBatchMatmulDispatch:
  """Emitters for the `linalg.batch_matmul` dispatch."""

  def __init__(self):
    self.mlir_dialect = MlirDialect.Linalg

    self.linalg_row_row_matmul_template = """
// Dispatch linalg.batch_matmul row-row layout 
func.func @${operation_name}_${compilation_info_name}(
  %lhs: tensor<${batch_size}x${problem_m}x${problem_k}x${datatype_lhs}>,
  %rhs: tensor<${batch_size}x${problem_k}x${problem_n}x${datatype_rhs}>) -> tensor<${batch_size}x${problem_m}x${problem_n}x${datatype_result}>
{
  %c0 = arith.constant 0.0 : ${datatype_result}
  %init = tensor.empty() : tensor<${batch_size}x${problem_m}x${problem_n}x${datatype_result}>
  %inital_result = linalg.fill ins(%c0 : ${datatype_result}) outs(%init : tensor<${batch_size}x${problem_m}x${problem_n}x${datatype_result}>) -> tensor<${batch_size}x${problem_m}x${problem_n}x${datatype_result}>
  %result = linalg.batch_matmul ${compilation_info_attribute} 
                     ins(%lhs, %rhs: tensor<${batch_size}x${problem_m}x${problem_k}x${datatype_lhs}>, tensor<${batch_size}x${problem_k}x${problem_n}x${datatype_rhs}>)
                     outs(%inital_result: tensor<${batch_size}x${problem_m}x${problem_n}x${datatype_result}>) -> tensor<${batch_size}x${problem_m}x${problem_n}x${datatype_result}>
  return %result : tensor<${batch_size}x${problem_m}x${problem_n}x${datatype_result}>
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
        'batch_size': str(dispatch.operation.batch_size),
        'problem_m': str(dispatch.operation.M),
        'problem_n': str(dispatch.operation.N),
        'problem_k': str(dispatch.operation.K),
        'datatype_lhs': DataTypeName[dispatch.operation.lhs.datatype],
        'datatype_rhs': DataTypeName[dispatch.operation.rhs.datatype],
        'datatype_result': DataTypeName[dispatch.operation.result.datatype],
        'compilation_info_name': dispatch.configuration.name()
    }

    return SubstituteTemplate(self.linalg_row_row_matmul_template, values)


class ReferenceBatchMatmulOperation:
  """Reference implementation for the batch matmul operation in numpy.
      ReferenceMatmulOperation class has the following responsibilities:
       1) Generates matmul operation inputs as np.array for a desired 
          distribution.
       2) Runs the matmul reference operation in np.matmul.
       3) Generates the matmul operation expected output as np.array.
       4) Additional, generate input and output filename strings.
  """

  def __init__(self, bmm_operation, dist_lhs, dist_rhs):
    self.bmm_operation = bmm_operation
    # Problem shape.
    self.batch_size = bmm_operation.batch_size
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
    self.filename_lhs = "batch_size{batch_size}xm{problem_m}xk{problem_k}_"\
      "{tensor_description}_{dist}_lhs.npy".format(
      batch_size=self.batch_size,
      problem_m=self.M,
      problem_k=self.K,
      tensor_description=self.bmm_operation.lhs.name(),
      dist=DistributionName[self.dist_lhs])

    # Filename for the right hand side input tensor.
    self.filename_rhs = "batch_size{batch_size}xk{problem_k}xn{problem_n}_"\
      "{tensor_description}_{dist}_rhs.npy".format(
      batch_size=self.batch_size,
      problem_k=self.K,
      problem_n=self.N,
      tensor_description=self.bmm_operation.rhs.name(),
      dist=DistributionName[self.dist_rhs])

    # Filename for the reference result tensor.
    self.reference_filename_result = "batch_size{batch_size}xm{problem_m}xn{problem_n}_"\
      "{tensor_description}_reference_result.npy".format(
      batch_size=self.batch_size,
      problem_m=self.M,
      problem_n=self.N,
      tensor_description=self.bmm_operation.result.name())

  # Generates input data, runs reference numpy.matmul, and save npy files to the output directory.
  def run_and_save(self, output_dir="."):

    # Generate the input data as np.array for the matmul operation.
    lhs_np_array = get_np_array(self.bmm_operation.lhs,
                                (self.batch_size, self.M, self.K),
                                self.dist_lhs)
    rhs_np_array = get_np_array(self.bmm_operation.rhs,
                                (self.batch_size, self.K, self.N),
                                self.dist_rhs)

    # Run the reference np.matmul and generate result np.array.
    result = np.matmul(lhs_np_array, rhs_np_array)

    # Save the input data as np.array for the matmul operation.
    np.save(os.path.join(output_dir, self.filename_lhs),\
             np.array(lhs_np_array, dtype = self.dtype_lhs))
    np.save(os.path.join(output_dir, self.filename_rhs),\
             np.array(rhs_np_array, dtype = self.dtype_rhs))

    # Save the expected result as an np.array.
    np.save(os.path.join(output_dir, self.reference_filename_result),\
             np.array(result, dtype = self.dtype_result))


###############################################################################
class BatchMatmulOperationLauncher:
  """Launches the compilation and execution of the matmul operation.
  BatchMatmulOperationLauncher class has the following responsibilities:
  """

  def __init__(self, args, operation):
    self.operation = operation

    # Variables from top-level argparse.
    self.generated_path = os.path.join(args.build_dir, 'generated',
                                       args.mlir_dialect)
    self.args = args
    self.benchmark_dispatch_repeat_count = args.batch_size
    self.batch_size = args.batch_size

    # Additional paths.
    self.operation_path = os.path.join(
        self.generated_path, OperationKindNames[operation.operation_kind],
        operation.name())
    self.source_mlir_file = os.path.join(self.operation_path,
                                         operation.name() + '.mlir')

    # path to iree-compile tool. (for compiling the input mlir file to vmfb)
    self.iree_compile_path = os.path.join(args.build_dir, 'tools',
                                          'iree-compile')

    # path to iree-benchmark-module tool. (for performance benchmarking and profiling)
    self.iree_benchmark_module_path = os.path.join(args.build_dir, 'tools',
                                                   'iree-benchmark-module')

    # path to iree-run-module tool. (for verification)
    self.iree_run_module_path = os.path.join(args.build_dir, 'tools',
                                             'iree-run-module')

    # output vmfb files for the operation.
    self.vmfb_verify_file = os.path.join(self.operation_path,
                                         self.operation.name() + '_verify.vmfb')
    self.vmfb_benchmark_file = os.path.join(
        self.operation_path,
        self.operation.name() + '_benchmark.vmfb')

  def compile(self, compilation_mode):
    """Compiles the matmul operation to a vmfb file for profiling."""

    benchmark_dispatch_repeat_count = self.benchmark_dispatch_repeat_count if compilation_mode == CompilationMode.Profile else 1
    vmfb_file = self.vmfb_benchmark_file if compilation_mode == CompilationMode.Profile else self.vmfb_verify_file

    # Base iree-compile commandline
    cmd = [self.iree_compile_path, self.source_mlir_file, "-o", f"{vmfb_file}"]

    # General compilation options
    cmd += [f"--iree-hal-target-backends={self.args.device}"]

    if self.args.device == "cuda":
      cmd += [f"--iree-hal-cuda-llvm-target-arch={self.args.cuda_arch}"]
    if self.args.split_k_slices != "":
      cmd += [f"--iree-flow-split-matmul-reduction={self.args.split_k_slices}"]
    if self.args.use_mma_sync:
      cmd += [f"--iree-codegen-llvmgpu-use-mma-sync"]
    if self.args.use_wmma:
      cmd += [f"--iree-codegen-llvmgpu-use-wmma"]

    # Compilation options for profiling
    cmd += [
        f"--iree-hal-benchmark-dispatch-repeat-count={benchmark_dispatch_repeat_count}"
    ]

    # Appends print ir options at the end of the command line.
    if self.args.mlir_print_ir_after_all:
      cmd += [f"--mlir-print-ir-after-all"]

    if not os.path.exists(vmfb_file) or self.args.force_compile:
      print(f">> Compilation command for "
            f"{CompilationModeNames[compilation_mode]} : {' '.join(cmd)}")
      #subprocess.check_output(cmd)
      compile_log_filename = f"{self.operation_path}/iree_compile_logs.mlir"
      with open(compile_log_filename, "w") as fp:
        subprocess.run(cmd, stderr=fp)

    elif self.args.verbose:
      print("Skipping compilation of matmul operation: " + vmfb_file +
            " since it already exists.")

  def verify(self, configuration):
    """Verifies the matmul operation with a given configuration."""
    # First compile the operation to a vmfb file.
    self.compile(CompilationMode.Verify)

    # Verify using random data distribution.
    # TODO 1) make input distribution configurable through command line.
    # TODO 2) make the reference run to check if reference npy files are present,
    #         then do not re-run the reference.
    reference_op = ReferenceBatchMatmulOperation(self.operation,\
                                         Distribution.Random,\
                                         Distribution.Random)

    lhs_npy_file = os.path.join(self.operation_path, reference_op.filename_lhs)
    rhs_npy_file = os.path.join(self.operation_path, reference_op.filename_rhs)
    expected_result_npy_file = os.path.join(
        self.operation_path, reference_op.reference_filename_result)

    # If the reference numpy do not exists, run the reference implementation
    # and generate npy files.
    if not os.path.exists(lhs_npy_file) or \
       not os.path.exists(rhs_npy_file) or \
       not os.path.exists(expected_result_npy_file):
      reference_op.run_and_save(self.operation_path)

    # Commandline `iree-run-module` for verification.
    cmd = [
        self.iree_run_module_path, f'--module={self.vmfb_verify_file}',
        f'--device={self.args.device}'
    ]

    # Operation-specific verification command-line.
    # TODO: abstract the operation-specific verification command-line out of verification.
    cmd.append(f'--function={self.operation.name()}_{configuration.name()}')
    cmd.append(f'--input=@{lhs_npy_file}')
    cmd.append(f'--input=@{rhs_npy_file}')
    cmd.append(f'--expected_output=@{expected_result_npy_file}')

    # Print the command if verbose.
    if self.args.verbose:
      print(">> Verification command: " + ' '.join(cmd))

    # Launch verification.
    cmd_output = subprocess.check_output(cmd, text=True)

    # Parse the verification output.
    m = re.search(r"\[(?P<verification_result>[a-zA-Z]+)\]", cmd_output)
    if m is None:
      raise Exception(
          "Failed to parse verification output by iree-run-module: " +
          cmd_output)
    verification_result = m.group('verification_result')

    if self.args.verbose or verification_result != "SUCCESS":
      print(cmd_output)

    return verification_result

  def profile(self, configuration):
    """Profiles the matmul operation with a given configuration."""
    # First compile the operation to a vmfb file.
    self.compile(CompilationMode.Profile)

    # Commandline `iree-benchmark-module` for profiling.
    cmd = [
        self.iree_benchmark_module_path, f'--module={self.vmfb_benchmark_file}',
        f'--device={self.args.device}'
    ]

    # Profiling specific flags.
    cmd += [f'--benchmark_repetitions={self.args.benchmark_repetitions}']
    cmd += [f'--batch_size={self.batch_size}']

    # Operation-specific profiling command-line.
    cmd += [f'--function={self.operation.name()}_{configuration.name()}']
    cmd += [f'--input={self.operation.lhs_npy_shape()}']
    cmd += [f'--input={self.operation.rhs_npy_shape()}']

    # Print the command if verbose.
    if self.args.verbose:
      print(">> Profiling command: " + ' '.join(cmd))

    # Launch profiling.
    cmd_output = subprocess.check_output(cmd,
                                         text=True,
                                         stderr=subprocess.STDOUT)

    # Parse the profiling output.
    m = re.search(r"real_time_median\s+(?P<runtime>\d+.\d+)\s+ms", cmd_output)
    if m is None:
      raise Exception("Failed to parse runtime from benchmark result: " +
                      cmd_output)
    runtime_in_ms = float(m.group('runtime'))
    return runtime_in_ms


##############################################################################
class CudaBatchMatmulGenerator:
  """Matmul dispatch generator class.
  Generates a list of pre-definied matmul operations with resonable tuning cofigurations. 
  The generator function are seperated based on the target backend and the data type.
  Please see example `MatmulGenerator._cuda_matmul_tensor_cores_f16` for cuda target 
  backend and f16 data type."""

  def __init__(self, args):

    self.args = args

    self.translation_infos = [
        #TranslationInfo.LLVMGPUMatmulTensorCore, # Tensor Core (WMMA)
        TranslationInfo.
        LLVMGPUMatmulTensorCoreMmaSync,  # Tensor Core (MMA.SYNC)
    ]

    self.bmm_shapes = [[16, 128, 128, 256]]

    # List of pre-definied matmul dispatch collections.
    self.dispatches_collection_list = []

  def _cuda_matmul_tensor_cores_f16(self):
    """Appends a list of matmul dispatches for GPU TensorCore F16 data type."""

    tile_descriptions = [
        TileDescription([64, 64, 64], 5, [64, 2, 1]),
        TileDescription([64, 64, 32], 10, [64, 2, 1]),
    ]

    # Create configuration list from the tile descriptions and translation infos.
    configuration_list = []

    for tile_description in tile_descriptions:
      for translation_info in self.translation_infos:
        configuration_list.append(
            BatchMatmulCompilationInfo(tile_description, translation_info))

    # Create dispatches collection for each problem shape with the configuration list.
    for bmm_shape in self.bmm_shapes:
      operation = BatchMatmulOperation(
        bmm_shape,\
        TensorDescription(DataType.f16, LayoutType.RowMajor), \
        TensorDescription(DataType.f16, LayoutType.RowMajor), \
        TensorDescription(DataType.f16, LayoutType.RowMajor))

      # Filter out configurations that are not supported by LLVM GPU CUDA backend.
      supported_configuration_list = []
      """
      supported_configuration_list = self._cuda_supported_configuration_list(
          operation, configuration_list)

      # Add default configuration if requested.
      if self.args.default_config:
        supported_configuration_list.append(
            BatchMatmulCompilationInfo([], [], CompilationConfigType.Default))
      """
      self.dispatches_collection_list.append(DispatchCollection(\
        operation, configuration_list))

  def generate(self):
    """Generates a list of batch matmul operations."""
    self._cuda_matmul_tensor_cores_f16()
    return self.dispatches_collection_list
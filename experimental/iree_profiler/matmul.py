import enum
import os.path
import shutil
import functools
import operator
import collections
import subprocess
from library import *


###############################################################################
# Data structure to describe a matrix multiplication operation including the
# shape, datatype, and layout of the operands and the result. This data structure
# is *independent* of the compilation strategy and tiling parameters, and only
# describes the parameter that changes the functionality of matmul operation.
###############################################################################
class MatmulOperation:
  #
  def __init__(self, problem_shape, lhs, rhs, result):
    # Initializes a matrix multiplication operation with the specified shape.
    # problem_shape: A tuple representing the matrix multiplication problem shape
    #                in the format (M, N, K), where M is the number of rows in the
    #                lhs matrix, N is the number of columns in the rhs matrix, and
    #                K is the number of columns in the lhs matrix and rows in the
    #                rhs matrix.
    # lhs: A TensorDescription object representing the left-hand-side matrix operand.
    # rhs: A TensorDescription object representing the right-hand-side matrix operand.
    # result: A TensorDescription object representing the result matrix operand.
    self.problem_shape = problem_shape  # Matmul problem shape [M, N, K]
    self.M = problem_shape[0]
    self.N = problem_shape[1]
    self.K = problem_shape[2]

    self.lhs = lhs  # TensorDescription
    self.rhs = rhs  # TensorDescription
    self.result = result  # TensorDescription
    self.operation_kind = OperationKind.Matmul

  # Procedurally generated name for the matmul operation. The name uniquely
  # identifies a matmul operation with matmul shape, lhs dataype and layout,
  # rhs datatype and layout, and result datatype and layout.
  def name(self):
    return "matmul_{m}x{n}x{k}_"\
      "{typeLhs}{layoutLhs}_"\
      "{typeRhs}{layoutRhs}_"\
      "{typeResult}{layoutResult}".format(
      m=self.problem_shape[0],
      n=self.problem_shape[1],
      k=self.problem_shape[2],
      typeLhs=DataTypeName[self.lhs.datatype],
      layoutLhs=ShortLayoutTypeName[self.lhs.layout],
      typeRhs=DataTypeName[self.rhs.datatype],
      layoutRhs=ShortLayoutTypeName[self.rhs.layout],
      typeResult=DataTypeName[self.result.datatype],
      layoutResult=ShortLayoutTypeName[self.result.layout]
    )

  # Returns the shape of the lhs numpy array as a string in the format "MxKxDataType".
  def lhs_npy_shape(self):
    return str(self.M) + "x" + str(self.K) + "x" + \
      str(DataTypeName[self.lhs.datatype])

  # Returns the shape of the rhs numpy array as a string in the format "KxNxDataType".
  def rhs_npy_shape(self):
    return str(self.K) + "x" + str(self.N) + "x" + \
      str(DataTypeName[self.rhs.datatype])

  # Returns the shape of the result numpy array as a string in the format "MxNxDataType".
  def result_npy_shape(self):
    return str(self.M) + "x" + str(self.N) + "x" + \
      str(DataTypeName[self.result.datatype])

  def bytes(self):
    bytes = (DataTypeSizeInBits[self.lhs.datatype] * self.M // 8) * self.K + \
            (DataTypeSizeInBits[self.rhs.datatype] * self.K // 8) * self.N + \
            (DataTypeSizeInBits[self.result.datatype] * self.M // 8) * self.N
    return bytes

  def flops(self):
    return 2 * self.M * self.N * self.K


###############################################################################
# Data structure to *only* describe the compilation strategy and the tiling
# parameters for a matrix multiplication operation that influences the
# performance of the compiledmatmul operation. This data structure should be
# independent of the matmul operation functionality, or any change in this data
# structure should not affect the functionality of the matmul operation.
###############################################################################
class MatmulCompilationInfo:
  #
  def __init__(self, tile_description, translation_info):
    self.tile_description = tile_description  # TileDescription
    self.translation_info = translation_info  # TranslationInfo

  #
  def name(self):
    return "tile_config_{tbm}x{tbn}_{tbk}x{stages}_{translation_info}".format(
        tbm=str(self.tile_description.threadblock_shape[0]),
        tbn=str(self.tile_description.threadblock_shape[1]),
        tbk=str(self.tile_description.threadblock_shape[2]),
        stages=str(self.tile_description.stages),
        translation_info=TranslationInfoName[self.translation_info])


###############################################################################
# Emitters for the matmul compilation info.
###############################################################################
class EmitMatmulCompilationInfo:

  def __init__(self):
    self.compilation_info_template = """
// matmul compilation info (tile configuration, translation info, workgroup size)
#${compilation_info_name} = #iree_codegen.compilation_info<
  lowering_config = <tile_sizes = [[${threadblock_shape_m}, ${threadblock_shape_n}, ${threadblock_shape_k}]]>,
  translation_info = <${translation_info} pipeline_depth = ${stages}>,
  workgroup_size = [${block_dim_x} : index, ${block_dim_y} : index, ${block_dim_z} : index]
>
"""

  #
  def emit(self, compilation_info):
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


###############################################################################
# MLIR Emitter for the matmul operation.
###############################################################################


# Emiter `linalg.matmul` operation.
class EmitLinalgMatmulOperation:

  def __init__(self):
    self.mlir_dialect = MlirDialect.Linalg

    self.linalg_row_row_matmul_template = """
// linalg.matmul operation row-row layout 
func.func @${operation_name}_${compilation_trait}(
  %lhs: tensor<${problem_m}x${problem_k}x${datatype_lhs}>,
  %rhs: tensor<${problem_k}x${problem_n}x${datatype_rhs}>) -> tensor<${problem_m}x${problem_n}x${datatype_result}>
{
  %c0 = arith.constant 0.0 : ${datatype_result}
  %init = tensor.empty() : tensor<${problem_m}x${problem_n}x${datatype_result}>
  %inital_result = linalg.fill ins(%c0 : ${datatype_result}) outs(%init : tensor<${problem_m}x${problem_n}x${datatype_result}>) -> tensor<${problem_m}x${problem_n}x${datatype_result}>
  %result = linalg.matmul {compilation_info = #${compilation_trait}} 
                     ins(%lhs, %rhs: tensor<${problem_m}x${problem_k}x${datatype_lhs}>, tensor<${problem_k}x${problem_n}x${datatype_rhs}>)
                     outs(%inital_result: tensor<${problem_m}x${problem_n}x${datatype_result}>) -> tensor<${problem_m}x${problem_n}x${datatype_result}>
  return %result : tensor<${problem_m}x${problem_n}x${datatype_result}>
}
"""

  # Emit the matmul operation in the MLIR dialect for a single compilation info.
  def emit(self, matmul_operation, compilation_info):
    values = {
        'operation_name': matmul_operation.name(),
        'problem_m': str(matmul_operation.problem_shape[0]),
        'problem_n': str(matmul_operation.problem_shape[1]),
        'problem_k': str(matmul_operation.problem_shape[2]),
        'datatype_lhs': DataTypeName[matmul_operation.lhs.datatype],
        'datatype_rhs': DataTypeName[matmul_operation.rhs.datatype],
        'datatype_result': DataTypeName[matmul_operation.result.datatype],
        'compilation_trait': compilation_info.name()
    }

    return SubstituteTemplate(self.linalg_row_row_matmul_template, values)


# Emit `mhlo.matmul` operation.
# TODO: Add support for testing lowering matmul op from other dialect.
class EmitMhloMatmulOperation:

  def __init__(self):
    self.mlir_dialect = MlirDialect.Mhlo

    self.linalg_row_row_matmul_template = """
// mhlo.matmul operation row-row layout
"""


###############################################################################
# Emitters for the matmul operation MLIR source files with different tuning
# configurations. The matmul operation MLIR is currently generated as
# `linalg.matmul` using, EmitLinalgMatmulOperation but can be trivially
# extended to other dialects.
###############################################################################
class EmitMatmulSourceMlir:

  def __init__(self, operation_path, operation, configuration_list):
    self.operation_path = operation_path
    self.operation = operation
    self.configuration_list = configuration_list

    self.operation_filepath = os.path.join(self.operation_path, \
                                           self.operation.name() + ".mlir")

  def __enter__(self):
    self.operation_file = open(self.operation_filepath, "w")
    self.operation_file.write('// Finename: ' + self.operation_filepath)

    # emit all the configuration attribute tags.
    for configuration in self.configuration_list:
      self.operation_file.write(EmitMatmulCompilationInfo().emit(configuration))

    return self

  def emit(self):
    # emit the matmul func.func dispatch (operation + configuration)
    for configuration in self.configuration_list:
      print("    Emitting matmul tuning parameters: " + configuration.name())
      self.operation_file.write(EmitLinalgMatmulOperation().emit(self.operation,\
                                                                 configuration))

  def __exit__(self, exc_type, exc_value, traceback):
    self.operation_file.close()


###############################################################################
# ReferenceMatmulOperation class has the following responsibilities:
# 1) Generates matmul operation inputs as np.array for a desired distribution.
# 2) Runs the matmul reference operation in np.matmul.
# 3) Generates the matmul operation expected output as np.array.
# 4) Additional, generate input and output filename strings.
###############################################################################
class ReferenceMatmulOperation:

  def __init__(self, matmul_operation, dist_lhs, dist_rhs):
    self.matmul_operation = matmul_operation
    # Problem shape.
    self.M = matmul_operation.problem_shape[0]
    self.N = matmul_operation.problem_shape[1]
    self.K = matmul_operation.problem_shape[2]

    # dtype for the input and result matrices.
    self.dtype_lhs = DataTypeNumPyTag[matmul_operation.lhs.datatype]
    self.dtype_rhs = DataTypeNumPyTag[matmul_operation.rhs.datatype]
    self.dtype_result = DataTypeNumPyTag[matmul_operation.result.datatype]

    # Distribution of the input tensors.
    self.dist_lhs = dist_lhs
    self.dist_rhs = dist_rhs

    # filename for the lhs, rhs, inital_result, and result matrices.
    self.filename_lhs = "m{problem_m}xk{problem_k}_"\
      "{tensor_description}_{dist}_lhs.npy".format(
      problem_m=str(self.M),
      problem_k=str(self.K),
      tensor_description=self.matmul_operation.lhs.name(),
      dist=DistributionName[self.dist_lhs])

    self.filename_rhs = "k{problem_k}xn{problem_n}_"\
      "{tensor_description}_{dist}_rhs.npy".format(
      problem_k=str(self.K),
      problem_n=str(self.N),
      tensor_description=self.matmul_operation.rhs.name(),
      dist=DistributionName[self.dist_rhs])

    self.reference_filename_result = "m{problem_m}xn{problem_n}_"\
      "{tensor_description}_reference_result.npy".format(
      problem_m=str(self.M),
      problem_n=str(self.N),
      tensor_description=self.matmul_operation.result.name())

  # Generates input data, runs reference numpy.matmul, and save npy files to the output directory.
  def run_and_save(self, output_dir="."):

    # Generate the input data as np.array for the matmul operation.
    lhs_np_array = get_np_array(self.matmul_operation.lhs, (self.M, self.K),
                                self.dist_lhs)
    rhs_np_array = get_np_array(self.matmul_operation.rhs, (self.K, self.N),
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
# MatmulOperationLauncher class has the following responsibilities:
# 1) Launches compilation of the matmul for a verification or profiling runs.
# 2) Launches the verification by running the IREE compiled matmul operation
#    and the python nmpy.matmul.
# 3) Launches the profiling by running the IREE compiled matmul operation.
###############################################################################
class MatmulOperationLauncher:

  def __init__(self, args, operation):
    self.operation = operation

    # Variables from top-level argparse.
    self.generated_path = os.path.join(args.build_dir, 'generated',
                                       args.mlir_dialect)
    self.device = args.device
    self.benchmark_dispatch_repeat_count = args.batch_size
    self.batch_size = args.batch_size
    self.benchmark_repetitions = args.benchmark_repetitions
    self.verbose = False if args.verbose in ['False', 'false', '0'] else True

    # Additional paths.
    self.matmul_path = os.path.join(self.generated_path, 'matmul')
    self.operation_path = os.path.join(self.matmul_path, operation.name())
    self.source_mlir_file = os.path.join(self.operation_path,
                                         operation.name() + '.mlir')

    # path to iree-compile tool. (for compiling the input mlir file to vmfb)
    self.iree_compile_path = os.path.join(args.build_dir, 'tools/iree-compile')
    self.force_compile = False if args.force_compile in ['False', 'false', '0'
                                                        ] else True

    # path to iree-benchmark-module tool. (for performance benchmarking and profiling)
    self.iree_benchmark_module_path = os.path.join(
        args.build_dir, 'tools/iree-benchmark-module')

    # path to iree-run-module tool. (for verification)
    self.iree_run_module_path = os.path.join(args.build_dir,
                                             'tools/iree-run-module')

    # output vmfb files for the operation.
    self.vmfb_verify_file = os.path.join(self.operation_path,
                                         self.operation.name() + '_verify.vmfb')
    self.vmfb_benchmark_file = os.path.join(
        self.operation_path,
        self.operation.name() + '_benchmark.vmfb')

  # Compile the matmul operation to a vmfb file for benchmarking.
  def compile(self, compilation_mode):

    cmd_template = \
    "${iree_compile_path} ${source_mlir_file} --iree-hal-target-backends=${device} "\
    "--iree-hal-cuda-llvm-target-arch=sm_80 --iree-hal-benchmark-dispatch-repeat-count=${benchmark-dispatch-repeat-count} "\
    "--iree-codegen-llvmgpu-use-mma-sync=true -o ${vmfb_file}"

    benchmark_dispatch_repeat_count = self.benchmark_dispatch_repeat_count if compilation_mode == CompilationMode.Benchmark else 1
    vmfb_file = self.vmfb_benchmark_file if compilation_mode == CompilationMode.Benchmark else self.vmfb_verify_file

    values = {
        'iree_compile_path': self.iree_compile_path,
        'source_mlir_file': self.source_mlir_file,
        'device': self.device,
        'benchmark-dispatch-repeat-count': str(benchmark_dispatch_repeat_count),
        'vmfb_file': vmfb_file
    }

    if not os.path.exists(vmfb_file) or self.force_compile:
      cmd = SubstituteTemplate(cmd_template, values)
      print(">> Compilation command for " +
            CompilationModeNames[compilation_mode] + " : " + cmd)
      subprocess.getoutput(cmd)

    else:
      if self.verbose:
        print("Skipping compilation of matmul operation: " + vmfb_file +
              " since it already exists.")

  # Verify a matmul operation with a given configuration.
  def verify(self, configuration):
    # First compile the operation to a vmfb file.
    self.compile(CompilationMode.Verify)

    # Template for the verification command using iree-run-module as the verification tool.
    cmd_templete = \
    "${iree_run_module_path} --module=${vmfb_file} --device=${device} --function=${function_name} "\
    "--input=@${lhs_npy_file} --input=@${rhs_npy_file} "\
    "--expected_output=@${expected_result_npy_file}"

    # Function name operation + configuration.
    function_name = self.operation.name() + "_" + configuration.name()

    # Verify using random data distribution.
    # TODO 1) make input distribution configurable through command line.
    # TODO 2) make the reference run to check if reference npy files are present,
    #         then do not re-run the reference.
    reference_op = ReferenceMatmulOperation(self.operation,\
                                         Distribution.Random,\
                                         Distribution.Random)

    lhs_npy_file = os.path.join(self.operation_path, reference_op.filename_lhs)
    rhs_npy_file = os.path.join(self.operation_path, reference_op.filename_rhs)
    expected_result_npy_file = os.path.join(
        self.operation_path, reference_op.reference_filename_result)

    # Run the reference implementation and generate npy files
    reference_op.run_and_save(self.operation_path)

    values = {
        'iree_run_module_path': self.iree_run_module_path,
        'vmfb_file': self.vmfb_verify_file,
        'device': self.device,
        'function_name': function_name,
        'lhs_npy_file': lhs_npy_file,
        'rhs_npy_file': rhs_npy_file,
        'expected_result_npy_file': expected_result_npy_file
    }
    # Create the verification command.
    cmd = SubstituteTemplate(cmd_templete, values)
    if self.verbose:
      print(">> Verification command: " + cmd)

    # Launch verification.
    cmd_output = subprocess.getoutput(cmd)

    # Parse the verification output.
    m = re.search(r"\[(?P<verification_result>[a-zA-Z]+)\]", cmd_output)
    if m is None:
      raise Exception(
          "Failed to parse verification output by iree-run-module: " +
          cmd_output)
    verification_result = m.group('verification_result')

    if self.verbose or verification_result != "SUCCESS":
      print(cmd_output)

    return verification_result

  # Profile a matmul operation given a tuning configuration.
  def profile(self, configuration):
    # First compile the operation to a vmfb file.
    self.compile(CompilationMode.Benchmark)

    # Template for the benchmarking command using iree-benchmark-module as the benchmarking tool.
    cmd_templete = \
    "${iree_benchmark_module_path} --module=${vmfb_file} --device=${device} "\
    "--benchmark_repetitions=${benchmark_repetitions} --batch_size=${batch_size} --function=${function_name} "\
    "--input=${lhs_npy_shape} --input=${rhs_npy_shape}"

    # Benchmark function name operation + configuration.
    function_name = self.operation.name() + "_" + configuration.name()

    # Benchmark using random npy array data distribution.
    lhs_npy_shape = self.operation.lhs_npy_shape()
    rhs_npy_shape = self.operation.rhs_npy_shape()

    values = {
        'iree_benchmark_module_path': self.iree_benchmark_module_path,
        'vmfb_file': self.vmfb_benchmark_file,
        'device': self.device,
        'benchmark_repetitions': str(self.benchmark_repetitions),
        'batch_size': str(self.batch_size),
        'function_name': function_name,
        'lhs_npy_shape': lhs_npy_shape,
        'rhs_npy_shape': rhs_npy_shape
    }

    cmd = SubstituteTemplate(cmd_templete, values)
    if self.verbose:
      print(">> Profiling command: " + cmd)
    cmd_output = subprocess.getoutput(cmd)
    m = re.search(r"real_time_median\s+(?P<runtime>\d+.\d+)\s+ms", cmd_output)
    if m is None:
      raise Exception("Failed to parse runtime from benchmark result: " +
                      cmd_output)
    runtime_in_ms = float(m.group('runtime'))
    return runtime_in_ms


###############################################################################
# Free functions to create a list of pre-backed matmul operations along with
# tuning cofigurations. The functions below seperated based on the target backend
# and the data type.
###############################################################################


# Matmul sizes and tuning configs for GPU TensorCore F16 data type.
def GpuMatmulTensorCoresF16(mainfest):

  # Matmul tuning configurations for LLVM GPU TensorCore(F16)
  tile_descriptions = [

      # Test tile that works for both wmma and mma.sync
      #TileDescription([128, 128, 32], 3, [64, 2, 1]),

      # Tiles for performance profiling `mma.sync.[f16/f32].f16.f16.[f16/f32]``
      TileDescription([128, 256, 32], 3, [128, 2, 1]),
      TileDescription([128, 128, 64], 4, [64, 2, 1]),
      TileDescription([128, 128, 32], 5, [64, 2, 1]),
      TileDescription([128, 64, 32], 5, [64, 2, 1]),
      TileDescription([64, 64, 64], 5, [64, 2, 1]),
      TileDescription([64, 64, 32], 10, [64, 2, 1]),
  ]

  # compilation info configuration list.
  configuration_list = []
  translation_info = TranslationInfo.LLVMGPUMatmulTensorCoreMmaSync
  for tile_description in tile_descriptions:
    configuration_list.append(
        MatmulCompilationInfo(tile_description, translation_info))

  # Matmul problems.
  problem_shapes = [
      #[128, 128, 256],
      [256, 256, 256],
      #[2560, 2560, 2560],
      #[1024, 512, 2048],
      #[3456, 1024, 2048]
  ]

  # Create matmul 'operation collection list' : [{operation -> [configurations]}].
  operations_collection_list = []

  for problem_shape in problem_shapes:
    operation = MatmulOperation(
      problem_shape,\
      TensorDescription(DataType.f16, LayoutType.RowMajor), \
      TensorDescription(DataType.f16, LayoutType.RowMajor), \
      TensorDescription(DataType.f16, LayoutType.RowMajor))

    mainfest.append_operation_collection(OperationCollection(\
      operation, configuration_list))


###############################################################################


# Matmul sizes and tuning configs for GPU TensorCore F16 data type.
def GpuMatmulTensorCoresF32(mainfest):

  # Matmul tuning configurations for LLVM GPU TensorCore(F32)
  tile_descriptions = [
      TileDescription([128, 256, 16], 3, [128, 2, 1]),
      #TileDescription([256, 128, 16], 3, [64, 4, 1]), # This tile does not iree-compile.
      TileDescription([128, 128, 16], 5, [64, 2, 1]),
      TileDescription([128, 128, 32], 4, [64, 2, 1]),
      TileDescription([64, 64, 64], 3, [64, 2, 1]),
  ]

  # compilation info configuration list.
  configuration_list = []
  translation_info = TranslationInfo.LLVMGPUMatmulTensorCoreMmaSync
  for tile_description in tile_descriptions:
    configuration_list.append(
        MatmulCompilationInfo(tile_description, translation_info))

  # Matmul problems.
  problem_shapes = [
      #[128, 128, 256],
      [256, 256, 256],
      #[1024, 512, 2048],
      #[3456, 1024, 2048]
  ]

  # Create matmul 'operation collection list' : [{operation -> [configurations]}].
  operations_collection_list = []

  for problem_shape in problem_shapes:
    operation = MatmulOperation(
      problem_shape,\
      TensorDescription(DataType.f32, LayoutType.RowMajor), \
      TensorDescription(DataType.f32, LayoutType.RowMajor), \
      TensorDescription(DataType.f32, LayoutType.RowMajor))

    mainfest.append_operation_collection(OperationCollection(\
      operation, configuration_list))


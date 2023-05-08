import enum
from enum import auto
import re
import numpy as np
from collections import namedtuple

###################################################################################################
# This file contains library of enumerations and classes used to build operation descritpions.
# The operation descriptions are used to generate MLIR source files, performance tuning configuration,
# reference implementations, and numpy input/output files.

# The file is organized as follows:
# 1. Enumerated `Type`s grouped together for categories, For e.g. [Arch]Type, [Data]Type etc.
# 2. Dictonaries `Names` mapping the enumeration values to their string names.
#    For e.g. [Arch]TypeNames, [Data]TypeNames etc.
# 3. `Tags` for each enumeration value to be used in the generated MLIR source files.
#    For e.g. [TranslationInfo]Tags
###################################################################################################


# Architecure types
###################################################################################################
class ArchType(enum.Enum):
  Cpu = auto()
  Gpu = auto()


ArchTypeNames = {
    ArchType.Cpu: "cpu",
    ArchType.Gpu: "gpu",
}


class GpuArchType(enum.Enum):
  nvptx = auto()
  rocm = auto()
  spirv = auto()


GpuArchTypeNames = {
    GpuArchType.nvptx: "nvptx",
    GpuArchType.rocm: "rocm",
    GpuArchType.spirv: "spirv",
}


# Operation kinds
###################################################################################################
class OperationKind(enum.Enum):
  Matmul = auto()
  Conv2d = auto()


OperationKindNames = {
    OperationKind.Matmul: 'matmul',
    OperationKind.Conv2d: 'conv2d'
}


# MLIR dialects
###################################################################################################
class MlirDialect(enum.Enum):
  Linalg = auto()
  Mhlo = auto()


MlirDialectNames = {
    MlirDialect.Linalg: 'linalg',
    MlirDialect.Mhlo: 'mhlo',
}


# Compilation modes (verification or benchmarking/profiling)
###################################################################################################
class CompilationMode(enum.Enum):
  Verify = auto()
  Profile = auto()


CompilationModeNames = {
    CompilationMode.Verify: 'verify',
    CompilationMode.Profile: 'profile',
}


class CompilationConfigType(enum.Enum):
  Default = auto()
  Custom = auto()


CompilationConfigTypeName = {
    CompilationConfigType.Default: 'default',
    CompilationConfigType.Custom: 'custom',
}


# Enumerations for data types and layouts
###################################################################################################
class DataType(enum.Enum):
  b1 = auto()
  u4 = auto()
  u8 = auto()
  u16 = auto()
  u32 = auto()
  u64 = auto()
  s4 = auto()
  s8 = auto()
  s16 = auto()
  s32 = auto()
  s64 = auto()
  e4m3 = auto()
  e5m2 = auto()
  f16 = auto()
  bf16 = auto()
  f32 = auto()
  tf32 = auto()
  f64 = auto()
  invalid = auto()


DataTypeName = {
    DataType.b1: "b1",
    DataType.u4: "u4",
    DataType.u8: "u8",
    DataType.u16: "u16",
    DataType.u32: "u32",
    DataType.u64: "u64",
    DataType.s4: "s4",
    DataType.s8: "s8",
    DataType.s16: "s16",
    DataType.s32: "s32",
    DataType.s64: "s64",
    DataType.e4m3: 'e4m3',
    DataType.e5m2: 'e5m2',
    DataType.f16: "f16",
    DataType.bf16: "bf16",
    DataType.f32: "f32",
    DataType.tf32: "tf32",
    DataType.f64: "f64",
}

DataTypeNumPyTag = {
    DataType.f16: np.float16,
    DataType.f32: np.float32,
}

DataTypeSizeInBits = {
    DataType.b1: 1,
    DataType.u4: 4,
    DataType.u8: 8,
    DataType.u16: 16,
    DataType.u32: 32,
    DataType.u64: 64,
    DataType.s4: 4,
    DataType.s8: 8,
    DataType.s16: 16,
    DataType.s32: 32,
    DataType.s64: 64,
    DataType.e4m3: 8,
    DataType.e5m2: 8,
    DataType.f16: 16,
    DataType.bf16: 16,
    DataType.f32: 32,
    DataType.tf32: 32,
    DataType.f64: 64,
}


class LayoutType(enum.Enum):
  ColumnMajor = auto()
  RowMajor = auto()
  NHWC = auto()
  NCWH = auto()


# cuBLAS/cuDNN layout type names convention is followed for the layout names.
# https://docs.nvidia.com/cuda/cublas/index.html#cublasoperation-t
ShortLayoutTypeName = {
    LayoutType.ColumnMajor: 'n',
    LayoutType.RowMajor: 't',
    LayoutType.NHWC: 'nhwc',
    LayoutType.NCWH: 'ncwh',
}


# Compilation pipelines/translation info.
###################################################################################################
class TranslationInfo(enum.Enum):
  LLVMGPUMatmulSIMT = auto()
  LLVMGPUMatmulTensorCore = auto()
  LLVMGPUMatmulTensorCoreMmaSync = auto()


TranslationInfoTag = {
    TranslationInfo.LLVMGPUMatmulSIMT:
        "LLVMGPUMatmulSIMT",
    TranslationInfo.LLVMGPUMatmulTensorCore:
        "LLVMGPUMatmulTensorCore",
    TranslationInfo.LLVMGPUMatmulTensorCoreMmaSync:
        "LLVMGPUMatmulTensorCoreMmaSync",
}

TranslationInfoName = {
    TranslationInfo.LLVMGPUMatmulSIMT: "simt",
    TranslationInfo.LLVMGPUMatmulTensorCore: "tensorcore_wmma",
    TranslationInfo.LLVMGPUMatmulTensorCoreMmaSync: "tensorcore_mma_sync",
}


# Distribution of values in a tensor.
###################################################################################################
class Distribution(enum.Enum):
  Empty = auto()
  Zeros = auto()
  Ones = auto()
  Sequential = auto()
  Identity = auto()
  Random = auto()


DistributionName = {
    Distribution.Empty: "empty",
    Distribution.Zeros: "zeros",
    Distribution.Ones: "ones",
    Distribution.Sequential: "seq",
    Distribution.Identity: "identity",
    Distribution.Random: "random",
}

###################################################################################################
# The next part of this file contains the data structures for describing a tensor, tiles etc that
# are built using the above enumerations. These data structures are used to create compose bigger
# data structures that describe an operation or a sequence of operations, along with compilation
# pipeling to form a collection of dispatches to profiled.
###################################################################################################


class TensorDescription:
  """A class for tensor description."""

  def __init__(self, datatype, layout):
    self.datatype = datatype
    self.layout = layout

  def name(self):
    return "%s%s" % (DataTypeName[self.datatype],
                     ShortLayoutTypeName[self.layout])


class TileDescription:
  """A class for tile description."""

  def __init__(self, threadblock_shape, stages, block_dim):
    self.threadblock_shape = threadblock_shape  # in number of elements in M, N, K
    self.stages = stages  # number of shared memory stages in tile K
    self.block_dim = block_dim  # block dimension in number of threads in x, y, z

  def name(self):
    return "%dx%d_%dx%d" % (self.threadblock_shape[0],
                            self.threadblock_shape[1],
                            self.threadblock_shape[2], self.stages)


###################################################################################################
# The following part contains utility functions for which are used by the profiler tool.
# These function may be moved as the need for create a proper structure for the
# functionality they provide becomes apparent and necessary as we move forward.
###################################################################################################
def get_np_array(tensor_description, shape, dist):
  """Returns a numpy array based on the distribution and shape."""
  # Fix the seed for reproducibility.
  np.random.seed(42)

  # Generate the numpy array based on the distribution.
  if dist == Distribution.Empty:
    return np.empty(shape)
  elif dist == Distribution.Zeros:
    return np.zeros(shape)
  elif dist == Distribution.Ones:
    return np.ones(shape)
  elif dist == Distribution.Sequential:
    return np.arange(np.prod(shape)).reshape(shape)
  elif dist == Distribution.Identity:
    return np.eye(shape[0], shape[1])
  elif dist == Distribution.Random:
    if tensor_description.datatype == DataType.s8:
      return np.random.randint(-2, 3, shape)
    elif tensor_description.datatype == DataType.u8:
      return np.random.randint(0, 4, shape)
    elif tensor_description.datatype == DataType.f16 or \
         tensor_description.datatype == DataType.bf16:
      return np.random.randint(-3, 4, shape)
    elif tensor_description.datatype == DataType.f32:
      return np.random.randint(-7, 8, shape)


###################################################################################################
def SubstituteTemplate(template, values):
  """Substitutes values into a template string."""
  text = template
  for key, value in values.items():
    regex = "\\$\\{%s\\}" % key
    newtext = re.sub(regex, value, text)
    text = newtext
  return text

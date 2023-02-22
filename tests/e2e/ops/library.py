import enum
from enum import auto as enum_auto
import re
import numpy as np
from collections import namedtuple
###################################################################################################
# This file contains library of enumerations and classes used to build operation descritpions.
# The operation descriptions are used to generate MLIR source files, performance tuning configuration,
# reference implementations, and numpy input/output files.
###################################################################################################

# Architecure types
###################################################################################################
class ArchType(enum.Enum):
  Cpu = enum_auto()
  Gpu = enum_auto()

#
ArchTypeNames = {
  ArchType.Cpu: "cpu",
  ArchType.Gpu: "gpu",
}

#
class GpuArchType(enum.Enum):
  nvptx = enum_auto()
  rocm = enum_auto()
  spirv = enum_auto()

#
GpuArchTypeNames = {
  GpuArchType.nvptx: "nvptx",
  GpuArchType.rocm: "rocm",
  GpuArchType.spirv: "spirv",
}
###################################################################################################

#
class OperationKind(enum.Enum):
  Matmul = enum_auto()
  Conv = enum_auto()       

#
OperationKindNames = {
  OperationKind.Matmul: 'matmul', 
  OperationKind.Conv: 'conv'  
}

class MlirDialect(enum.Enum):
  Linalg = enum_auto()
  Mhlo = enum_auto()

#
MlirDialectNames = {
  MlirDialect.Linalg: 'linalg',
  MlirDialect.Mhlo: 'mhlo',
}

class CompilationMode (enum.Enum):
  Verify = enum_auto()
  Benchmark = enum_auto()

#
CompilationModeNames = {
  CompilationMode.Verify: 'verify',
  CompilationMode.Benchmark: 'benchmark',
}
###################################################################################################


# Enumerations for data types and layouts
###################################################################################################
class DataType(enum.Enum):
  b1 = enum_auto()
  u4 = enum_auto()
  u8 = enum_auto()
  u16 = enum_auto()
  u32 = enum_auto()
  u64 = enum_auto()
  s4 = enum_auto()
  s8 = enum_auto()
  s16 = enum_auto()
  s32 = enum_auto()
  s64 = enum_auto()
  e4m3 = enum_auto()
  e5m2 = enum_auto()
  f16 = enum_auto()
  bf16 = enum_auto()
  f32 = enum_auto()
  tf32 = enum_auto()
  f64 = enum_auto()
  invalid = enum_auto()


#
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
  ColumnMajor = enum_auto()
  RowMajor = enum_auto()
  NHWC = enum_auto()
  NCWH = enum_auto()


# cuBLAS layout type names convenctions ()
ShortLayoutTypeName = {
  LayoutType.ColumnMajor: 'n',
  LayoutType.RowMajor: 't',
  LayoutType.NHWC: 'nhwc',
  LayoutType.NCWH: 'ncwh',
}


###################################################################################################
# Data structures for describing a tensor, tiles and fundametal operation types 
###################################################################################################
class TensorDescription:
  def __init__(self, datatype, layout):
    self.datatype = datatype
    self.layout = layout

  def name(self):
    return "%s%s" % (DataTypeName[self.datatype], ShortLayoutTypeName[self.layout])


###################################################################################################
class TileDescription:

  def __init__(self, threadblock_shape, stages, block_dim):
    self.threadblock_shape = threadblock_shape # in number of elements in M, N, K
    self.stages = stages                       # number of shared memory stages in tile K
    self.block_dim = block_dim                 # block dimension in number of threads in x, y, z


  def name(self):
    return "%dx%d_%dx%d" % (self.threadblock_shape[0], self.threadblock_shape[1], self.threadblock_shape[2], self.stages)


###################################################################################################
# Compilation pipelines/translation info
###################################################################################################
class TranslationInfo(enum.Enum):
  LLVMGPUMatmulTensorCore = enum_auto()
  LLVMGPUMatmulSIMT = enum_auto()

#
TranslationInfoTag = {
  TranslationInfo.LLVMGPUMatmulTensorCore: "LLVMGPUMatmulTensorCore",
  TranslationInfo.LLVMGPUMatmulSIMT: "LLVMGPUMatmulSIMT",
}

#
TranslationInfoName = {
  TranslationInfo.LLVMGPUMatmulTensorCore: "tensorcore",
  TranslationInfo.LLVMGPUMatmulSIMT: "simt",
}

# A collection of indpendent dispatches. A dispatch is operation description + configuration.
# A single mlir file is generated per operation enumerating multiple dispatches, one for each
# configuration in the list. Total number of dispatches by one instaces of OperationCollection
# is len(configuration_list).
OperationCollection = namedtuple('OperationCollection', ['operation', 'configuration_list'])


###################################################################################################
# Enumerations for npy files
###################################################################################################
class Distribution(enum.Enum):
  Empty = enum_auto()
  Zeros = enum_auto()
  Ones = enum_auto()
  Sequential = enum_auto()
  Identity = enum_auto()
  Random = enum_auto()

#
DistributionName = {
  Distribution.Empty: "empty",
  Distribution.Zeros: "zeros",
  Distribution.Ones: "ones",
  Distribution.Sequential: "seq",
  Distribution.Identity: "identity",
  Distribution.Random: "random",
}

# Helper function to generate a npy file name from a distribution and shape.
def get_np_array(tensor_description, shape, dist):
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
      return np.random.randint(-2, 2, shape)
    elif tensor_description.datatype == DataType.u8:
      return np.random.randint(0, 4, shape)
    elif tensor_description.datatype == DataType.f16 or \
         tensor_description.datatype == DataType.bf16:
      return np.random.randint(-4, 4, shape)
    elif tensor_description.datatype == DataType.f32:
      return np.random.randint(-8, 8, shape)

###################################################################################################

# Helper function to substitute values into a template string.
def SubstituteTemplate(template, values):
  text = template
  changed = True
  while changed:
    changed = False
    for key, value in values.items():
      regex = "\\$\\{%s\\}" % key
      newtext = re.sub(regex, value, text)
      if newtext != text:
        changed = True
      text = newtext
  return text

###################################################################################################

import enum
import os.path
import shutil

from library import *
from matmul import *
#
class Manifest:
    
  #
  def __init__(self, args):
    self.args = args
    # operation_kind -> [operation_collection]
    self.operations = {}

    self.kernel_filter = ''
  
  # Appends a single instance of OperationCollection to the manifest.
  def append_operation_collection(self, operation_collection):

    operation_kind = operation_collection.operation.operation_kind
    if operation_kind not in self.operations.keys():
      self.operations[operation_kind] = []

    self.operations[operation_kind].append(operation_collection)

  # Appends a list of OperationCollection to the manifest.
  def append(self, operation_collection_list):
    for operation_collection in operation_collection_list:
      self.append_operation_collection(operation_collection)


  # Emits the operations in the mainfest to the build directory as MLIR source files.
  # The operations are emitted in the dialect specified by the mlir_dialect flag.
  def emit(self, mlir_dialect = MlirDialect.Linalg):
    mlir_source_emitter = {
      OperationKind.Matmul : EmitMatmulSourceMlir,
    }

    generated_path = os.path.join(self.args.build_dir, 'generated', MlirDialectNames[mlir_dialect])

    if os.path.exists(generated_path):
      shutil.rmtree(generated_path)

    os.makedirs(generated_path)

    # For each operation_kind create a directory and emit the operations with 
    # all the configurations in the configuration_list into their seperate directories.
    for operation_kind, operation_collection_list in self.operations.items():

      operation_kind_path = os.path.join(generated_path, OperationKindNames[operation_kind])
      
      if os.path.exists(operation_kind_path):
        shutil.rmtree(operation_kind_path)
      os.makedirs(operation_kind_path)

      for operation_collection in operation_collection_list:
        
        operation_path = os.path.join(operation_kind_path, operation_collection.operation.name())

        if os.path.exists(operation_path):
          shutil.rmtree(operation_path)
        os.makedirs(operation_path)

        with mlir_source_emitter[operation_kind](operation_path,\
                            operation_collection.operation,\
                            operation_collection.configuration_list)\
                            as mlir_source:
          
          print(">> Generating MLIR operation: " + operation_collection.operation.name())
          # Emit mlir source file for the operation_collection.operation with all the configurations
          mlir_source.emit()
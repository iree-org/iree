import enum
import os.path
import shutil

from library import *
from matmul import *


# Manifest class collects all the operations and configurations and emits \
# them to the `generated` directory.
class Manifest:
  #
  def __init__(self, args):
    self.args = args

    # operation_kind -> [operation_collection]
    self.operations = {}

    # For operation kind-based filtering of dispatches.
    self.operation_kind_enabled = []

    # For name-based filtering of dispatches.
    self.dispatch_names = []
    self.ignore_dispatch_names = []

    if args.operation_kind == 'all':
      self.operation_kind_enabled = []
    else:
      operations_kind_list = [
          OperationKind.Matmul,
          #OperationKind.Conv2d
      ]
      self.operation_kind_enabled = [x for x in operations_kind_list\
                                     if OperationKindNames[x] in\
                                     args.operation_kind.split(',')]

    if args.dispatches == 'all':
      self.dispatch_names = []
    else:
      self.dispatch_names = [x for x in args.dispatches.split(',') if x != '']


# Returns true if all substrings appear in the haystack in order

  def _filter_string_matches(self, filter_string, haystack):

    substrings = filter_string.split('*')
    for sub in substrings:
      idx = haystack.find(sub)
      if idx < 0:
        return False
      haystack = haystack[idx + len(sub):]
    return True

  # Filters the dispatches (operation, specific configuration) \
  # in the manifest based various criteria.
  def filter(self, operation, configuration):

    # If no filter is enabled then return True.
    enabled = True

    # If operation_kind filter is enabled and the \
    # operation_kind in not in the enabled list, return False.
    if len(self.operation_kind_enabled) and \
      operation.operation_kind not in self.operation_kind_enabled:
      enabled = False

    # If dispatch name-based filter regex is enabled match the \
    # dispatch name (operation+configuration) against all regexs \
    # in self.dispatch_names.
    if len(self.dispatch_names):
      name = operation.name() + '_' + configuration.name()
      enabled = False

      # compare against each regex included in self.dispatch_names.
      for substr_to_match in self.dispatch_names:
        if self._filter_string_matches(substr_to_match, name):
          enabled = True
          break

    # Return the result of the filter.
    return enabled

  # Appends a single instance of OperationCollection to the manifest.
  def append_operation_collection(self, operation_collection):

    operation_kind = operation_collection.operation.operation_kind
    if operation_kind not in self.operations.keys():
      self.operations[operation_kind] = []

    filtered_operation_collection = OperationCollection(
        operation_collection.operation, [])
    for configuration in operation_collection.configuration_list:
      if self.filter(operation_collection.operation, configuration):
        filtered_operation_collection.configuration_list.append(configuration)

    # Only append the filtered_operation_collection if it has an unfiltered configuration.
    if len(filtered_operation_collection.configuration_list):
      self.operations[operation_kind].append(filtered_operation_collection)

  # Appends a list of OperationCollection to the manifest.
  def append(self, operation_collection_list):
    for operation_collection in operation_collection_list:
      self.append_operation_collection(operation_collection)

  # Emits the operations in the Manifest to the build directory as MLIR source files.
  # The operations are emitted in the dialect specified by the mlir_dialect flag.
  def emit(self, mlir_dialect=MlirDialect.Linalg):
    mlir_source_emitter = {
        OperationKind.Matmul: EmitMatmulSourceMlir,
        #OperationKind.Conv2d : EmitConv2dSourceMlir, TODO: Add conv2d
    }

    generated_path = os.path.join(self.args.build_dir, 'generated',
                                  MlirDialectNames[mlir_dialect])

    if os.path.exists(generated_path):
      shutil.rmtree(generated_path)

    os.makedirs(generated_path)

    # For each operation_kind create a directory and emit the operations with
    # all the configurations in the configuration_list into their seperate directories.
    for operation_kind, operation_collection_list in self.operations.items():

      operation_kind_path = os.path.join(generated_path,
                                         OperationKindNames[operation_kind])

      if os.path.exists(operation_kind_path):
        shutil.rmtree(operation_kind_path)
      os.makedirs(operation_kind_path)

      for operation_collection in operation_collection_list:

        operation_path = os.path.join(operation_kind_path,
                                      operation_collection.operation.name())

        if os.path.exists(operation_path):
          shutil.rmtree(operation_path)
        os.makedirs(operation_path)

        with mlir_source_emitter[operation_kind](operation_path,\
                            operation_collection.operation,\
                            operation_collection.configuration_list)\
                            as mlir_source:

          print(">> Generating MLIR operation: " +
                operation_collection.operation.name())
          # Emit mlir source file for the operation_collection.operation with all the configurations
          mlir_source.emit()

import enum
import os.path
import shutil

from library import *
from matmul import *


class Manifest:
  """Manifest collects, filters, and stores dispatches in a data structure.
     Manifest organizes the dispatches in a dictionary of `OperationKind` 
     to a list of `DispatchCollection`.

     OperationKind -> [DispatchCollection]
  """

  def __init__(self, args):
    self.args = args
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

  def _filter_string_matches(self, filter_string, haystack):
    """Returns true if all substrings appear in the haystack in order"""
    substrings = filter_string.split('*')
    for sub in substrings:
      idx = haystack.find(sub)
      if idx < 0:
        return False
      haystack = haystack[idx + len(sub):]
    return True

  def filter(self, dispatch):
    """Filters Dispatche based various criteria."""

    # Get the operation and configuration from the dispatch.
    operation = dispatch.operation
    configuration = dispatch.configuration

    # If the operation is not in the enabled list, return False.
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
      name = dispatch.name()
      enabled = False

      # compare against each regex included in self.dispatch_names.
      for substr_to_match in self.dispatch_names:
        if self._filter_string_matches(substr_to_match, name):
          enabled = True
          break

    # Return the result of the filter.
    return enabled

  def append_dispatch_collection(self, dispatch_collection):
    """Appends one instance of DispatchCollection to the manifest."""
    operation_kind = dispatch_collection.operation.operation_kind
    if operation_kind not in self.operations.keys():
      self.operations[operation_kind] = []

    # Get all the dispatches from the dispatch_collection.
    dispatches = dispatch_collection.get_dispatches()

    # Filter dispatches based on the filter criteria.
    filtered_dispatch_collection = DispatchCollection(
        dispatch_collection.operation, [])
    for dispatch in dispatches:
      if self.filter(dispatch):
        filtered_dispatch_collection.append(dispatch)

    # Only append the filtered_dispatch_collection if it has an unfiltered configuration.
    if len(filtered_dispatch_collection.configuration_list):
      self.operations[operation_kind].append(filtered_dispatch_collection)

  def append(self, dispatch_collection_list):
    """Appends one instance of DispatchCollection to the manifest."""
    for dispatch_collection in dispatch_collection_list:
      self.append_dispatch_collection(dispatch_collection)

  def emit(self, mlir_dialect=MlirDialect.Linalg):
    """Emits the operations in the Manifest to the build directory as MLIR source files.
        The operations are emitted in the dialect specified by the `mlir_dialect` flag.
    """
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
    for operation_kind, dispatch_collection_list in self.operations.items():

      operation_kind_path = os.path.join(generated_path,
                                         OperationKindNames[operation_kind])

      # If the directory with generated mlir already exists, delete it and create a new one.
      if os.path.exists(operation_kind_path):
        shutil.rmtree(operation_kind_path)
      os.makedirs(operation_kind_path)

      for dispatch_collection in dispatch_collection_list:

        operation_path = os.path.join(operation_kind_path,
                                      dispatch_collection.operation.name())

        if os.path.exists(operation_path):
          shutil.rmtree(operation_path)
        os.makedirs(operation_path)

        with mlir_source_emitter[operation_kind](operation_path, dispatch_collection)\
                            as mlir_source:

          print(">> Generating MLIR operation: " +
                dispatch_collection.operation.name())
          # Emit mlir source file for the dispatch_collection.operation with all the configurations
          mlir_source.emit()

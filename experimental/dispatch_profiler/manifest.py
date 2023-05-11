# Copyright 2022 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import enum, shutil, pickle
from library import *
from matmul import *
from batch_matmul import *
from split_k_matmul import *
from pathlib import Path


###############################################################################
class EmitSourceMLIR:
  """Emitters for the operation MLIR source files."""

  def __init__(self, operation_path, dispatch_collection):
    self.operation_path = operation_path
    self.dispatch_collection = dispatch_collection
    self.operation = dispatch_collection.operation
    self.operation_kind = self.operation.operation_kind
    self.configuration_list = dispatch_collection.configuration_list
    self.operation_filepath = self.operation_path.joinpath(
        self.operation.name()).with_suffix(".mlir")

    mlir_configuration_emitter = {
        OperationKind.Matmul: EmitMatmulCompilationInfo,
        OperationKind.SplitkMatmul: EmitMatmulCompilationInfo,
        OperationKind.BatchMatmul: EmitMatmulCompilationInfo,
    }
    self.configuration_emitter = mlir_configuration_emitter[
        self.operation_kind]()

    mlir_dispatch_emitter = {
        OperationKind.Matmul: EmitLinalgMatmulDispatch,
        OperationKind.SplitkMatmul: EmitLinalgMatmulDispatch,
        OperationKind.BatchMatmul: EmitLinalgBatchMatmulDispatch,
    }
    self.dispatch_emitter = mlir_dispatch_emitter[self.operation_kind]()

  def __enter__(self):
    self.operation_file = open(self.operation_filepath, "w")
    self.operation_file.write(f'// Finename: {self.operation_filepath}')

    # Emit all the configuration attribute tags.
    for configuration in self.configuration_list:
      self.operation_file.write(self.configuration_emitter.emit(configuration))
    return self

  def emit(self):
    """Emit the op func.func for each dispatch (operation + configuration)"""
    for dispatch in self.dispatch_collection.get_dispatches():
      print(
          f"    Emitting tuning configuration : {dispatch.configuration.name()}"
      )
      self.operation_file.write(self.dispatch_emitter.emit(dispatch))

  def __exit__(self, exc_type, exc_value, traceback):
    self.operation_file.close()


###############################################################################
class Manifest:
  """Manifest collects, filters, and stores dispatches in a data structure.
     Manifest organizes the dispatches in a dictionary. 
     Usage:
      1. Create a manifest object with the command line arguments.
      2(a). Generate dispatches, append them in the manifest, and 
            serialize them into a file.
      2(b). Load dispatches from a serialized file.
      
      ```python
      # generator.py usage:
      manifest = Manifest(args)
      manifest.initialize()

      # compile.py or profile.py usage:
      manifest = Manifest(args)
      manifest.load()
      ```
  """

  def __init__(self, args):
    self.args = args

    # Dictionary of operation kind to a list of dispatch collections. We
    # initialize the dictionary during the generation of dispatches and
    # serialize it to a file. The serialized file is used to load the
    # dispatches for compilation and profiling.
    # Datatype: OperationKind -> [DispatchCollection]
    self.dispatch_collection_map = {}

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
          OperationKind.SplitkMatmul,
          OperationKind.BatchMatmul,
      ]
      self.operation_kind_enabled = [
          x for x in operations_kind_list
          if OperationKindNames[x] in args.operation_kind.split(',')
      ]

    if args.dispatches == 'all':
      self.dispatch_names = []
    else:
      self.dispatch_names = [x for x in args.dispatches.split(',') if x != '']

    # Paths to the generated directory (e.g. `./generated/linalg`).
    self.generated_path = Path(self.args.generated_dir, 'generated',
                               self.args.mlir_dialect)

    # Create the directories in self.generated_path, if it does not exist.
    if not self.generated_path.exists():
      self.generated_path.mkdir(parents=True, exist_ok=True)

    # Path to the serialized file.
    self.serialized_file_path = self.generated_path.joinpath(
        'serialized_file.pkl')

  def _filter_string_matches(self, filter_string, haystack):
    """Returns true if all substrings appear in the haystack in order"""
    substrings = filter_string.split('*')
    for sub in substrings:
      idx = haystack.find(sub)
      if idx < 0:
        return False
      haystack = haystack[idx + len(sub):]
    return True

  def is_enabled(self, dispatch):
    """Rerturns true if pass through filters based various criteria."""

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
    if operation_kind not in self.dispatch_collection_map.keys():
      self.dispatch_collection_map[operation_kind] = []

    # Get all the dispatches from the dispatch_collection.
    dispatches = dispatch_collection.get_dispatches()

    # Filter dispatches based on the filter criteria.
    filtered_dispatch_collection = DispatchCollection(
        dispatch_collection.operation, [])
    for dispatch in dispatches:
      if self.is_enabled(dispatch):
        filtered_dispatch_collection.append(dispatch)

    # Only append the filtered_dispatch_collection if it has an unfiltered configuration.
    if len(filtered_dispatch_collection.configuration_list):
      self.dispatch_collection_map[operation_kind].append(
          filtered_dispatch_collection)

  def append(self, dispatch_collection_list):
    """Appends one instance of DispatchCollection to the manifest."""
    for dispatch_collection in dispatch_collection_list:
      self.append_dispatch_collection(dispatch_collection)

  def initialize(self):
    """Initialize the mainfest object by generating dispatches for supported operations."""
    self.append(CudaMatmulGenerator(self.args).generate())
    self.append(CudaSplitKMatmulGenerator(self.args).generate())
    self.append(CudaBatchMatmulGenerator(self.args).generate())

    # Serialize the initialized mainfest state.
    self.dump()

  def dump(self):
    """Serialize (dump) the self.dispatch_collection_map to a pickle file."""
    with open(self.serialized_file_path, 'wb') as f:
      pickle.dump(self.dispatch_collection_map, f)

  def load(self):
    """Deserialize (load) the self.dispatch_collection_map from a pickle file."""
    if not self.serialized_file_path.exists():
      raise ValueError(f"Could not find : {self.serialized_file_path}")

    with open(self.serialized_file_path, 'rb') as load_file:
      self.dispatch_collection_map = pickle.load(load_file)

  def emit(self):
    """Emits the operations in the Manifest to the build directory as MLIR source files.
        The operations are emitted in the dialect specified by the `mlir_dialect` flag.
    """

    # For each operation_kind create a directory and emit the operations with
    # all the configurations in the configuration_list into their seperate directories.
    for operation_kind, dispatch_collection_list\
      in self.dispatch_collection_map.items():

      operation_kind_path = self.generated_path.joinpath(
          OperationKindNames[operation_kind])

      # If the operation_kind_path does not exists, create it.
      if not operation_kind_path.exists():
        operation_kind_path.mkdir(parents=True, exist_ok=True)

      for dispatch_collection in dispatch_collection_list:

        operation_path = operation_kind_path.joinpath(
            dispatch_collection.operation.name())

        # If the operation_path does not exists, create it.
        if not operation_path.exists():
          operation_path.mkdir()

        with EmitSourceMLIR(operation_path,
                            dispatch_collection) as emit_mlir_source:
          mlir_file_path = operation_path.joinpath(
              dispatch_collection.operation.name()).with_suffix('.mlir')
          print(f"[Generating]: {mlir_file_path}")

          # Emit mlir source file for the dispatch_collection.operation with all the configurations
          emit_mlir_source.emit()

from library import *


################################################################################
class Dispatch:
  """
  Dispatch: A combination of an operation and a configuration is launched by 
    the dispatch profiler for verification and performance profiling. Note that 
    a dispatch is not a MLIR operation it is binary executable that is launched 
    by the profiler. Additionaly, the goal of the tool is to also profile the 
    performance of the fusions and a dispatch for fusion is a combination of 
    multiple operations glued together and compiled into a single dispatch.
  """

  def __init__(self, operation, configuration):
    self.operation = operation
    self.configuration = configuration
    self.is_fused_dispatch = False

  def name(self):
    return self.operation.name() + '_' + self.configuration.name()


################################################################################
class DispatchCollection:
  """
  DispatchCollection: A collection of dispatches that only vary in their 
    configurations but not in their operations. For example, a collection 
    of matmul dispatches with different tile sizes. 
    
    The idea is that we can emit a single MLIR file for all the dispatches 
    in the collection and compile with single run of iree-compile and them 
    into a single executable
  """

  def __init__(self, operation, configuration_list):
    self.operation = operation
    self.configuration_list = configuration_list

  def get_dispatches(self):
    """Returns a list of dispatches in the collection."""
    dispatches = []
    for configuration in self.configuration_list:
      dispatches.append(Dispatch(self.operation, configuration))
    return dispatches

  def append(self, dispatch):
    """Appends a dispatch to the collection."""
    if dispatch.operation != self.operation:
      print(self.operation.name())
      print(dispatch.operation.name())
      raise ValueError("operation (%s) does not match the "\
                       "dispatch collection operation name (%s)."\
                       % (self.operation.name(), dispatch.operation.name()))
    self.configuration_list.append(dispatch.configuration)

  def num_of_dispatches(self):
    """Returns number of dispatches in the collection."""
    return len(self.configuration_list)

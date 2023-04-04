import csv
import os
import numpy as np
from collections import namedtuple


class PerformanceResult:
  """Performance result of a single run."""

  def __init__(self, operation, configuration, verification_result, runtime):
    self.operation = operation
    self.configuration = configuration
    self.verification_result = verification_result
    self.bytes = operation.bytes()
    self.flops = operation.flops()
    self.runtime = runtime  # in milliseconds
    self.gflops = float(self.flops) / self.runtime / 1.0e6

  def print(self):
    """Prints the performance result to the console."""
    runtime = (str(self.runtime) if self.runtime != -1.0 else 'Not profiled')
    gflops = (str(float(round(self.gflops, 2)))
              if self.runtime != -1.0 else 'Not profiled')

    print('---------------------------------------------------------------- ')
    print('Dispatch      : %s' %
          "_".join([self.operation.name(),
                    self.configuration.name()]))
    print('Provider      : %s' % "IREE Codegen")
    print('Operation     : %s' % self.operation.name())
    print('Configuration : %s' % self.configuration.name())
    print('Verification  : %s' % self.verification_result)
    print('Bytes         : %d' % self.bytes)
    print('Flops         : %d' % self.flops)
    print('Runtime(ms)   : %s' % runtime)
    print('GFLOP/s       : %s' % gflops)

  def create_dict_entry(self):
    """Returns a dictionary with the performance result."""
    runtime = self.runtime if self.runtime != -1.0 else ''
    gflops = (float(round(self.gflops, 2))
              if self.runtime != -1.0 else 'Not run')
    return {
        'Provider': 'IREE Codegen',
        'Operation': self.operation.name(),
        'Configuration': self.configuration.name(),
        'Verification': self.verification_result,
        'Bytes': self.bytes,
        'Flops': self.flops,
        'Runtime(ms)': runtime,
        'GFLOP/s': gflops,
    }


class PerformanceReport:
  """Performance report class is used to store the performance results of multiple runs.
  The report can be written to a csv file."""

  def __init__(self, args):
    self.args = args

    # Data members extracted from the args.
    self.output_file_path = args.output

    # List of PerformanceResult.
    self.perf_result_vector = []

    # Additional tags to add to the csv report file. \
    # Useful for generating pivot tables.
    self.tags = []
    if args.tags != '':
      self.tags = args.tags.split(',')

    # If the args.output set, open the file and write the header.
    if self.output_file_path != '':
      open_mode = 'a' if self.args.append else 'w'
      self.csv_file = open(self.output_file_path, open_mode)

      # Create and write the header.
      common_header = ['Provider', 'Operation', 'Configuration']
      performance_header = [
          'Verification', 'Bytes', 'Flops', 'Runtime(ms)', 'GFLOP/s'
      ]
      csv_header = common_header + performance_header

      # If tags are present, add the tags.keys() to the csv header.
      if len(self.tags):
        tag_header = [tag.split(':')[0] for tag in self.tags]
        csv_header = tag_header + csv_header

      # Create the csv dictionary writer.
      self.csv_writer = csv.DictWriter(self.csv_file, fieldnames=csv_header)

      # Write the header if the file is being created.
      if open_mode == 'w':
        self.csv_writer.writeheader()

  def __del__(self):
    """If the args.output set, close the file."""
    if self.output_file_path != '':
      print('Writing performance report to %s' % self.output_file_path)
      self.csv_file.close()

  def append_perf_result(self, performance_result):
    """Appends a performance result to the report. 
    Additionaly, if args.output set, write the csv_row entry."""
    self.perf_result_vector.append(performance_result)

    if self.output_file_path != '':
      # Create the row entries for performance result.
      csv_dict_row = performance_result.create_dict_entry()

      # Create the row entries for tags.
      for tag in self.tags:
        tag_key, tag_value = tag.split(':')
        csv_dict_row[tag_key] = tag_value

      # Write the row.
      self.csv_writer.writerow(csv_dict_row)

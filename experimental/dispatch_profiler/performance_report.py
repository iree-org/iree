import csv
import os
import numpy as np
from collections import namedtuple


# Performance results class is used to store the performance results of a single run.
class PerformanceResult:

  def __init__(self, operation, configuration, verification_result, runtime):
    self.operation = operation
    self.configuration = configuration
    self.verification_result = verification_result
    self.bytes = operation.bytes()
    self.flops = operation.flops()
    self.runtime = runtime  # in milliseconds
    self.gflops = float(self.flops) / self.runtime / 1.0e6

  # Prints the performance result to the console.
  def print(self):

    runtime = (str(self.runtime) if self.runtime != -1.0 else 'Not run')
    gflops = (str(float(round(self.gflops, 2)))
              if self.runtime != -1.0 else 'Not run')

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

  # Returns a dictionary with the performance result. Used for csv writing.
  def create_dict_entry(self):
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


# Performance report class is used to store the performance results of multiple runs as
# a report. The report can be written to a csv file.
class PerformanceReport:
  #
  def __init__(self, args):
    self.args = args

    # Data members extracted from the args.
    self.output_file_path = args.output
    self.append = True if args.append in ['True', 'true', '1'] else False

    # List of PerformanceResult.
    self.perf_result_vector = []

    # Additional tags to add to the csv report file. \
    # Useful for generating pivot tables.
    self.tags = []
    if args.tags != '':
      self.tags = args.tags.split(',')

  # Appends a performance result to the report.
  def append_perf_result(self, performance_result):
    self.perf_result_vector.append(performance_result)

  # Writes the performance report to a csv file.
  def write_csv(self):
    open_mode = 'a' if self.append else 'w'

    with open(self.output_file_path, open_mode) as csv_file:
      # Create the csv header.
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
      csv_writer = csv.DictWriter(csv_file, fieldnames=csv_header)

      # Write the header if the file is being created.
      if open_mode == 'w':
        csv_writer.writeheader()

      # Write the performance results.
      for perf_result in self.perf_result_vector:
        # Create the row entries for performance result.
        csv_dict_row = perf_result.create_dict_entry()

        # Create the row entries for tags.
        for tag in self.tags:
          tag_key, tag_value = tag.split(':')
          csv_dict_row[tag_key] = tag_value

        # Write the row.
        csv_writer.writerow(csv_dict_row)

      print('Writing performance report to %s' % self.output_file_path)

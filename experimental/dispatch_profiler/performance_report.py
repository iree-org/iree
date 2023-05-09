# Copyright 2023 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import csv, textwrap
import numpy as np
from collections import namedtuple
from pathlib import Path


class PerformanceResult:
  """Performance result of a single run."""

  def __init__(self, operation, configuration, verification_result, runtime):
    self.operation = operation
    self.configuration = configuration
    self.verification_result = verification_result
    self.runtime = runtime  # in milliseconds
    self.gflops = float(self.operation.flops()) / self.runtime / 1.0e6

  def print(self):
    """Prints the performance result to the console."""
    runtime = (str(self.runtime) if self.runtime != -1.0 else 'Not profiled')
    gflops = (str(float(round(self.gflops, 2)))
              if self.runtime != -1.0 else 'Not profiled')

    print('---------------------------------------------------------------- ')
    print(
        f'Dispatch      : {"_".join([self.operation.name(), self.configuration.name()])}'
    )
    print(f'Provider      : IREE Codegen')
    print(f'OpKind        : {self.operation.operation_kind}')
    print(f'Operation     : {self.operation.name()}')
    print(f'Configuration : {self.configuration.name()}')
    # Operation specific arguments.
    arg_str = ' '.join([
        f'--{key}={value}'
        for key, value in self.operation.get_argument_dict().items()
    ])
    wrapped_arg_str = textwrap.fill(arg_str,
                                    width=80,
                                    subsequent_indent='                ')
    print(f'Arguments     : {wrapped_arg_str}')
    print(f'Verification  : {self.verification_result}')
    print(f'Runtime(ms)   : {runtime}')
    print(f'GFLOPs        : {gflops}')

  def get_dict_entry(self):
    """Returns a dictionary with the performance result."""
    runtime = self.runtime if self.runtime != -1.0 else ''
    gflops = (float(round(self.gflops, 2))
              if self.runtime != -1.0 else 'Not run')
    dict_entry = {
        'Provider': 'IREE Codegen',
        'Verification': self.verification_result,
        'Runtime(ms)': runtime,
        'GFLOPs': gflops,
    }

    # Add the operation specific arguments.
    dict_entry.update(self.operation.get_dict_entry())

    # Add the configuration specific arguments.
    dict_entry.update(self.configuration.get_dict_entry())

    return dict_entry


class PerformanceReport:
  """Performance report class is used to store the performance results of multiple runs.
  The report can be written to a csv file."""

  def __init__(self, args):
    self.args = args

    # Data members extracted from the args.
    self.output_file_path = None
    if args.output != '':
      self.output_file_path = Path(args.output)

    # List of PerformanceResult.
    self.perf_result_vector = []

    # Additional tags to add to the csv report file. \
    # Useful for generating pivot tables.
    self.tags = []
    if args.tags != '':
      self.tags = args.tags.split(',')

    # Boolen to check if the header is written to the csv file.
    self.is_header_written = False

    # If the args.output set, open the file and write the header.
    self.open_mode = 'a' if self.args.append else 'w'
    if self.output_file_path:
      self.csv_file = open(self.output_file_path, self.open_mode)

  def __del__(self):
    """If the args.output set, close the file."""
    if self.output_file_path:
      print('Writing performance report to %s' % self.output_file_path)
      self.csv_file.close()

  def write_csv_header(self, operation, configuration):
    """Write the header to the csv file."""

    # Create and write the header.
    operation_specific_header = list(operation.get_dict_entry().keys())
    configuration_specific_header = list(configuration.get_dict_entry().keys())
    performance_header = ['Verification', 'Runtime(ms)', 'GFLOPs']
    csv_header = operation_specific_header + configuration_specific_header + performance_header
    csv_header = ['Provider'] + csv_header

    # If tags are present, add the tags.keys() to the begining of the csv header.
    if len(self.tags):
      tag_header = [tag.split(':')[0] for tag in self.tags]
      csv_header = tag_header + csv_header

    # Create the csv dictionary writer.
    self.csv_writer = csv.DictWriter(self.csv_file, fieldnames=csv_header)

    # Write the header if the file is being created.
    if self.open_mode == 'w':
      self.csv_writer.writeheader()

  def append_perf_result(self, performance_result):
    """Appends a performance result to the report. 
    Additionaly, if args.output set, write the csv_row entry."""
    self.perf_result_vector.append(performance_result)

    if self.output_file_path:
      # Write the header if not written.
      if not self.is_header_written:
        self.write_csv_header(performance_result.operation,
                              performance_result.configuration)
        self.is_header_written = True

      # Create the row entries for performance result.
      csv_dict_row = performance_result.get_dict_entry()

      # Create the row entries for tags.
      for tag in self.tags:
        tag_key, tag_value = tag.split(':')
        csv_dict_row[tag_key] = tag_value

      # Write the row.
      self.csv_writer.writerow(csv_dict_row)

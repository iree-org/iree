# Copyright 2022 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import abc

from common.benchmark_command import BenchmarkCommand


class BenchmarkCommandFactory(abc.ABC):
  """ An abstract factory that generates commands depending on config.
  Args:
    device: Currently 'desktop' or 'mobile' are supported.
    driver: Currently 'cpu' or 'gpu' are supported.
  Returns:
    An array containing `BenchmarkCommand` objects.
  """

  @abc.abstractmethod
  def generate_benchmark_commands(self, device: str,
                                  driver: str) -> list[BenchmarkCommand]:
    pass

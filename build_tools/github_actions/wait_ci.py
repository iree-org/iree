#!/usr/bin/env python3

# Copyright 2023 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""TODO"""

import os
import requests


def main():
  github_token = os.environ["GITHUB_TOKEN"]
  commit_sha = os.environ["GITHUB_SHA"]
  print("Done")


if __name__ == "__main__":
  main()

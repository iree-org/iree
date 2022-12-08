## Copyright 2022 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Represents the directory structure of common artifacts."""

import pathlib
import urllib.parse

from e2e_test_framework.definitions import common_definitions

MODEL_ARTIFACTS_SUB_ROOT = pathlib.PurePath("model")
# Archive extensions used to pack models.
ARCHIVE_FILE_EXTENSIONS = [".tar", ".gz"]


def get_model_path(model: common_definitions.Model,
                   root_dir_path: pathlib.PurePath = pathlib.PurePath()):
  """Returns the path of an model artifact.
  
  Args:
    model: the model.
    root_dir_path: path of the root artifact directory, on which the returned
      path will be based.
  Returns:
    Path of the model artifact.
  """
  model_url = urllib.parse.urlparse(model.source_url)
  # Drop the archive extensions.
  file_exts = pathlib.PurePath(model_url.path).suffixes
  while len(file_exts) > 0 and file_exts[-1] in ARCHIVE_FILE_EXTENSIONS:
    file_exts.pop()
  model_ext = "".join(file_exts)

  # Model path: <model_artifacts_root>/<model_id>_<model_name><model_ext>
  return (root_dir_path / MODEL_ARTIFACTS_SUB_ROOT /
          f"{model.id}_{model.name}{model_ext}")

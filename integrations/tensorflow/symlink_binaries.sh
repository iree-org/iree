# Copyright 2021 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# Symlinks built binaries from the bazel-bin/ directory into the corresponding
# python packages.

set -e

cd "$(dirname $0)"
if [ -f bazel-bin/iree_tf_compiler/iree-import-tf ]; then
  ln -sf $PWD/bazel-bin/iree_tf_compiler/iree-import-tf python_projects/iree_tf/iree/tools/tf/
fi
if [ -f bazel-bin/iree_tf_compiler/iree-import-tflite ]; then
  ln -sf $PWD/bazel-bin/iree_tf_compiler/iree-import-tflite python_projects/iree_tflite/iree/tools/tflite/
fi
if [ -f bazel-bin/iree_tf_compiler/iree-import-xla ]; then
  ln -sf $PWD/bazel-bin/iree_tf_compiler/iree-import-xla python_projects/iree_xla/iree/tools/xla/
fi

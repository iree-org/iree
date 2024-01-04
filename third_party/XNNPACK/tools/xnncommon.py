#!/usr/bin/env python
# Copyright 2019 Google LLC
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Local changes (IREE specific, vs XNNPACK upstream):
# This copy only contains what is needed by xngen.py.

import codecs
import os

def overwrite_if_changed(filepath, content):
  txt_changed = True
  if os.path.exists(filepath):
    with codecs.open(filepath, "r", encoding="utf-8") as output_file:
      txt_changed = output_file.read() != content
  if txt_changed:
    with codecs.open(filepath, "w", encoding="utf-8") as output_file:
      output_file.write(content)

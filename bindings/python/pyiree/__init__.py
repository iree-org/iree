"""Module init for the python bindings."""

# Copyright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# pylint: disable=invalid-name
# pylint: disable=g-import-not-at-top


def __import_submodules():
  """Force a relative sub-module import.

  Python makes this hard when the package is at the top and this works
  around it, ensuring that submodules are loaded into the pyiree namespace.
  """
  import importlib

  def import_rel(m):
    abs_m = __package__ + "." + m
    importlib.import_module(abs_m)

  import_rel("binding")


__import_submodules()

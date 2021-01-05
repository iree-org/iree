# Copyright 2020 Google LLC
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

# This configure script wraps a normal CMake configure of the project for use
# in CI pipelines, doing some sanity checks, discovery and work-arounding.

# This future is needed to print Python2 EOL message
from __future__ import print_function
import sys
if sys.version_info < (3,):
  print("Python 2 has reached end-of-life and is no longer supported.")
  sys.exit(-1)
if sys.platform == 'win32' and sys.maxsize.bit_length() == 31:
  print(
      "32-bit Windows Python runtime is not supported. Please switch to 64-bit Python."
  )
  sys.exit(-1)

import importlib
import json
import os
import platform
import subprocess
import sysconfig
import tempfile

is_windows = platform.system() == 'Windows'


def report(*args):
  print('--', *args)


def get_setting(varname, default_value):
  value = os.environ.get(varname)
  if value is None:
    return default_value
  return value


def get_bool_setting(varname, default_value):
  value = get_setting(varname, default_value)
  if value is True or value is False:
    return value
  return value == '' or value == 'ON' or value == '1'


def which(thefile):
  path = os.environ.get("PATH", os.defpath).split(os.pathsep)
  for d in path:
    fname = os.path.join(d, thefile)
    fnames = [fname]
    if sys.platform == 'win32':
      exts = os.environ.get('PATHEXT', '').split(os.pathsep)
      fnames += [fname + ext for ext in exts]
    for name in fnames:
      if os.access(name, os.F_OK | os.X_OK) and not os.path.isdir(name):
        return name
  return None


def use_tool_path(toolname, varname=None):
  if not varname: varname = toolname.upper()
  value = get_setting(f'USE_{varname}', 'ON')
  if value.upper() == 'OFF':
    return None
  if value.upper() == 'ON' or value == '':
    return which(toolname)
  if os.access(value, os.F_OK | os.X_OK) and not os.path.isdir(value):
    return value


repo_root = os.path.abspath(
    get_setting('REPO_DIR', os.path.join(os.path.dirname(__file__), '..',
                                         '..')))

################################################################################
# Figure out where we are and where we are going.
################################################################################
repo_root = os.path.abspath(
    get_setting('REPO_DIR', os.path.join(os.path.dirname(__file__), '..',
                                         '..')))
report(f'Using REPO_DIR = {repo_root}')

################################################################################
# Build deps.
################################################################################
requirements_file = os.path.join(repo_root, 'bindings', 'python',
                                 'build_requirements.txt')

report('Installing python build requirements...')
subprocess.check_call(
    [sys.executable, '-m', 'pip', 'install', '-r', requirements_file])

################################################################################
# Load version_info.json
################################################################################


def load_version_info():
  with open(os.path.join(repo_root, 'version_info.json'), 'rt') as f:
    return json.load(f)


try:
  version_info = load_version_info()
except FileNotFoundError:
  report('version_info.json found')
  version_info = {}

################################################################################
# CMake configure.
################################################################################

cmake_args = [
    f'-S{repo_root}',
    f'-DPython3_EXECUTABLE:FILEPATH={sys.executable}',
    f'-DPython3_INCLUDE_DIR:PATH={sysconfig.get_path("include")}',
]

### HACK: Add a Python3_LIBRARY because cmake needs it, but it legitimately
### does not exist on manylinux (or any linux static python).
# Need to explicitly tell cmake about the python library.
python_libdir = sysconfig.get_config_var('LIBDIR')
python_library = sysconfig.get_config_var('LIBRARY')
if python_libdir and not os.path.isabs(python_library):
  python_library = os.path.join(python_libdir, python_library)

# On manylinux, python is a static build, which should be fine, but CMake
# disagrees. Fake it by letting it see a library that will never be needed.
if python_library and not os.path.exists(python_library):
  python_libdir = os.path.join(tempfile.gettempdir(), 'fake_python', 'lib')
  os.makedirs(python_libdir, exist_ok=True)
  python_library = os.path.join(python_libdir,
                                sysconfig.get_config_var('LIBRARY'))
  with open(python_library, 'wb') as f:
    pass

if python_library:
  cmake_args.append(f'-DPython3_LIBRARY:PATH={python_library}')

### Detect cmake.
use_cmake = use_tool_path('cmake') or 'cmake'

### Detect generator.
if use_tool_path('ninja'):
  report('Using ninja')
  cmake_args.append('-GNinja')
elif is_windows:
  cmake_args.extend(['-G', 'NMake Makefiles'])

# Detect other build tools.
use_ccache = use_tool_path('ccache')
if use_ccache:
  report(f'Using ccache {use_ccache}')
  cmake_args.append(f'-DCMAKE_CXX_COMPILER_LAUNCHER={use_ccache}')

# Clang
use_clang = use_tool_path('clang')
if use_clang:
  report(f'Using clang {use_clang}')
  cmake_args.append(f'-DCMAKE_C_COMPILER={use_clang}')
use_clangcpp = use_tool_path('clang++', 'CLANGCPP')
if use_clangcpp:
  report(f'Using clang++ {use_clangcpp}')
  cmake_args.append(f'-DCMAKE_CXX_COMPILER={use_clangcpp}')

# LLD
use_lld = use_tool_path('lld')
if not is_windows and use_lld:
  report(f'Using linker {use_lld}')
  cmake_args.append('-DIREE_ENABLE_LLD=ON')

cmake_args.extend(sys.argv[1:])
report(f'Running cmake (generate): {" ".join(cmake_args)}')
subprocess.check_call([use_cmake] + cmake_args)

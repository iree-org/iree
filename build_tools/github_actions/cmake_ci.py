# Copyright 2020 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

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


def display_help():
  print('Syntax: python build_tools/cmake/cmake_ci.py [--install|--build] ...')
  print('If neither --install or --build are the first argument, then it is ')
  print('assumed to be a generate invocation')


mode = 'generate'
if len(sys.argv) < 2:
  display_help()
  sys.exit(1)
if sys.argv[1] == '--install':
  mode = 'install'
elif sys.argv[1] == '--build':
  mode = 'build'


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
  if not varname:
    varname = toolname.upper()
  value = get_setting(f'USE_{varname}', 'ON')
  if value.upper() == 'OFF':
    return None
  if value.upper() == 'ON' or value == '':
    return which(toolname)
  if os.access(value, os.F_OK | os.X_OK) and not os.path.isdir(value):
    return value


### Detect cmake.
use_cmake = use_tool_path('cmake') or 'cmake'
cmake_command_prefix = [use_cmake]
cmake_environ = os.environ


def cmake_commandline(args):
  return cmake_command_prefix + args


if is_windows:
  # Bazel needs msys bash and TensorFlow will melt down and cry if it finds
  # system bash. Because, of course it will.
  # Note that we don't set this as a CMake option because it may have spaces
  # in the path, use backslashes or various other things that get corrupted
  # in the five or six layers of shoddy string transformations between here
  # and where it gets used.
  bash_exe = which('bash')
  report('Found Windows bash:', bash_exe)
  report('NOTE: If the above is system32 bash and you are using bazel to build '
         'TensorFlow, you are going to have a bad time. Suggest being explicit '
         'adding the correct directory to your path. I\'m really sorry. '
         'I didn\'t make this mess... just the messenger')
  report(f'Full path = {os.environ.get("PATH")}')


def invoke_generate():
  ##############################################################################
  # Figure out where we are and where we are going.
  ##############################################################################
  repo_root = os.path.abspath(
      get_setting('REPO_DIR', os.path.join(os.path.dirname(__file__), '..',
                                           '..')))
  report(f'Using REPO_DIR = {repo_root}')

  ##############################################################################
  # Load version_info.json
  ##############################################################################

  def load_version_info():
    with open(os.path.join(repo_root, 'version_info.json'), 'rt') as f:
      return json.load(f)

  try:
    version_info = load_version_info()
  except FileNotFoundError:
    report('version_info.json found')
    version_info = {}

  ##############################################################################
  # CMake configure.
  ##############################################################################

  cmake_args = [
      f'-S{repo_root}',
      f'-DPython3_EXECUTABLE:FILEPATH={sys.executable}',
      # The old python package settings should not be needed, but since there
      # can be configuration races between packages that use both mechanisms,
      # be explicit.
      f'-DPYTHON_EXECUTABLE:FILEPATH={sys.executable}',
      f'-DPython3_INCLUDE_DIR:PATH={sysconfig.get_path("include")}',
      f'-DPYTHON_INCLUDE_DIR:PATH={sysconfig.get_path("include")}',
      f'-DIREE_RELEASE_PACKAGE_SUFFIX:STRING={version_info.get("package-suffix") or ""}',
      f'-DIREE_RELEASE_VERSION:STRING={version_info.get("package-version") or "0.0.1a1"}',
      f'-DIREE_RELEASE_REVISION:STRING={version_info.get("iree-revision") or "HEAD"}',
  ]

  ### Detect generator.
  if use_tool_path('ninja'):
    report('Using ninja')
    cmake_args.append('-GNinja')
  elif is_windows:
    cmake_args.extend(['-G', 'NMake Makefiles'])

  # Detect other build tools.
  use_ccache = use_tool_path('ccache')
  if not is_windows and use_ccache:
    report(f'Using ccache {use_ccache}')
    cmake_args.append(f'-DCMAKE_CXX_COMPILER_LAUNCHER={use_ccache}')

  # Clang
  use_clang = use_tool_path('clang')
  if not is_windows and use_clang:
    report(f'Using clang {use_clang}')
    cmake_args.append(f'-DCMAKE_C_COMPILER={use_clang}')
  use_clangcpp = use_tool_path('clang++', 'CLANGCPP')
  if not is_windows and use_clangcpp:
    report(f'Using clang++ {use_clangcpp}')
    cmake_args.append(f'-DCMAKE_CXX_COMPILER={use_clangcpp}')

  # LLD
  use_lld = use_tool_path('lld')
  if not is_windows and use_lld:
    report(f'Using linker {use_lld}')
    cmake_args.append('-DIREE_ENABLE_LLD=ON')

  cmake_args.extend(sys.argv[1:])
  report(f'Running cmake (generate): {" ".join(cmake_args)}')
  subprocess.check_call(cmake_commandline(cmake_args), env=cmake_environ)


# Select which mode.
if mode == 'generate':
  invoke_generate()
else:
  # Just pass-through.
  cmake_args = cmake_commandline(sys.argv[1:])
  report('Invoke CMake:', ' '.join(cmake_args))
  subprocess.check_call(cmake_args, env=cmake_environ)

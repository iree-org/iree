# Copyright 2020 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import os
import platform
import re
import sys


def detect_unix_platform_config(bazelrc):
  # This is hoaky. Ideally, bazel had any kind of rational way of selecting
  # options from within its environment (key word: "rational"), but sadly, it
  # is unintelligible to mere mortals. Why should a build system have a way for
  # people to condition their build options on what compiler they are using
  # (without descending down the hole of deciphering what a Bazel toolchain is)?
  # All I want to do is set a couple of project specific warning options!

  if platform.system() == "Darwin":
    print(f"build --config=macos_clang", file=bazelrc)
    print(f"build:release --config=macos_clang_release", file=bazelrc)
  else:

    # If the user specified a CXX environment var, bazel will later respect that,
    # so we just see if it says "clang".
    cxx = os.environ.get("CXX")
    cc = os.environ.get("CC")
    if (cxx is not None and cc is None) or (cxx is None and cc is not None):
      print("WARNING: Only one of CXX or CC is set, which can confuse bazel. "
            "Recommend: set both appropriately (or none)")
    if cc is not None and cxx is not None:
      # Persist the variables.
      print(f"build --action_env CC=\"{cc}\"", file=bazelrc)
      print(f"build --action_env CXX=\"{cxx}\"", file=bazelrc)
    else:
      print(
          "WARNING: CC and CXX are not set, which can cause mismatches between "
          "flag configurations and compiler. Recommend setting them explicitly."
      )

    if cxx is not None and "clang" in cxx:
      print(
          f"Choosing generic_clang config because CXX is set to clang ({cxx})")
      print(f"build --config=generic_clang", file=bazelrc)
      print(f"build:release --config=generic_clang_release", file=bazelrc)
    else:
      print(f"Choosing generic_gcc config by default because no CXX set or "
            f"not recognized as clang ({cxx})")
      print(f"build --config=generic_gcc", file=bazelrc)
      print(f"build:release --config=generic_gcc_release", file=bazelrc)


def write_platform(bazelrc):
  if platform.system() == "Windows":
    print(f"build --config=msvc", file=bazelrc)
    print(f"build:release --config=msvc_release", file=bazelrc)
  else:
    detect_unix_platform_config(bazelrc)


# The code below has been borrowed from https://github.com/tensorflow/tensorflow/blob/master/configure.py
# to configure ANDROID environment variables when building with --config=android_arm64.
_DEFAULT_PROMPT_ASK_ATTEMPTS = 10
_SUPPORTED_ANDROID_NDK_VERSIONS = [
    10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21
]


class UserInputError(Exception):
  pass


def is_windows():
  return platform.system() == 'Windows'


def is_linux():
  return platform.system() == 'Linux'


def is_macos():
  return platform.system() == 'Darwin'


def write_to_bazelrc(line, file_path):
  with open(file_path, 'a') as f:
    f.write(line + '\n')


def get_input(question):
  try:
    try:
      answer = input(question)
    except NameError:
      answer = input(question)  # pylint: disable=bad-builtin
  except EOFError:
    answer = ''
  return answer


def write_action_env_to_bazelrc(var_name, var, file_path):
  write_to_bazelrc('build --action_env {}="{}"'.format(var_name, str(var)),
                   file_path)


def get_from_env_or_user_or_default(environ_cp, var_name, ask_for_var,
                                    var_default):
  """Get var_name either from env, or user or default.
  If var_name has been set as environment variable, use the preset value, else
  ask for user input. If no input is provided, the default is used.
  Args:
    environ_cp: copy of the os.environ.
    var_name: string for name of environment variable, e.g. "TF_NEED_CUDA".
    ask_for_var: string for how to ask for user input.
    var_default: default value string.
  Returns:
    string value for var_name
  """
  var = environ_cp.get(var_name)
  if not var:
    var = get_input(ask_for_var)
    print('\n')
  if not var:
    var = var_default
  return var


def prompt_loop_or_load_from_env(environ_cp,
                                 var_name,
                                 var_default,
                                 ask_for_var,
                                 check_success,
                                 error_msg,
                                 suppress_default_error=False,
                                 resolve_symlinks=False,
                                 n_ask_attempts=_DEFAULT_PROMPT_ASK_ATTEMPTS):
  """Loop over user prompts for an ENV param until receiving a valid response.
  For the env param var_name, read from the environment or verify user input
  until receiving valid input. When done, set var_name in the environ_cp to its
  new value.
  Args:
    environ_cp: (Dict) copy of the os.environ.
    var_name: (String) string for name of environment variable, e.g. "TF_MYVAR".
    var_default: (String) default value string.
    ask_for_var: (String) string for how to ask for user input.
    check_success: (Function) function that takes one argument and returns a
      boolean. Should return True if the value provided is considered valid. May
      contain a complex error message if error_msg does not provide enough
      information. In that case, set suppress_default_error to True.
    error_msg: (String) String with one and only one '%s'. Formatted with each
      invalid response upon check_success(input) failure.
    suppress_default_error: (Bool) Suppress the above error message in favor of
      one from the check_success function.
    resolve_symlinks: (Bool) Translate symbolic links into the real filepath.
    n_ask_attempts: (Integer) Number of times to query for valid input before
      raising an error and quitting.
  Returns:
    [String] The value of var_name after querying for input.
  Raises:
    UserInputError: if a query has been attempted n_ask_attempts times without
      success, assume that the user has made a scripting error, and will
      continue to provide invalid input. Raise the error to avoid infinitely
      looping.
  """
  default = environ_cp.get(var_name) or var_default
  full_query = '%s [Default is %s]: ' % (
      ask_for_var,
      default,
  )

  for _ in range(n_ask_attempts):
    val = get_from_env_or_user_or_default(environ_cp, var_name, full_query,
                                          default)
    if check_success(val):
      break
    if not suppress_default_error:
      print(error_msg % val)
    environ_cp[var_name] = ''
  else:
    raise UserInputError('Invalid %s setting was provided %d times in a row. '
                         'Assuming to be a scripting mistake.' %
                         (var_name, n_ask_attempts))

  if resolve_symlinks and os.path.islink(val):
    val = os.path.realpath(val)
  environ_cp[var_name] = val
  return val


def cygpath(path):
  """Convert path from posix to windows."""
  return os.path.abspath(path).replace('\\', '/')


def create_android_ndk_rule(environ_cp, save_path):
  """Set ANDROID_NDK_HOME and write Android NDK WORKSPACE rule."""
  if is_windows():
    default_ndk_path = cygpath('%s/Android/Sdk/ndk-bundle' %
                               environ_cp['APPDATA'])
  elif is_macos():
    default_ndk_path = '%s/library/Android/Sdk/ndk-bundle' % environ_cp['HOME']
  else:
    default_ndk_path = '%s/Android/Sdk/ndk-bundle' % environ_cp['HOME']

  def valid_ndk_path(path):
    return (os.path.exists(path) and
            os.path.exists(os.path.join(path, 'source.properties')))

  android_ndk_home_path = prompt_loop_or_load_from_env(
      environ_cp,
      var_name='ANDROID_NDK_HOME',
      var_default=default_ndk_path,
      ask_for_var='Please specify the home path of the Android NDK to use.',
      check_success=valid_ndk_path,
      error_msg=('The path %s or its child file "source.properties" '
                 'does not exist.'))
  write_action_env_to_bazelrc('ANDROID_NDK_HOME', android_ndk_home_path,
                              save_path)
  write_action_env_to_bazelrc(
      'ANDROID_NDK_API_LEVEL',
      get_ndk_api_level(environ_cp, android_ndk_home_path), save_path)


def create_android_sdk_rule(environ_cp, save_path):
  """Set Android variables and write Android SDK WORKSPACE rule."""
  if is_windows():
    default_sdk_path = cygpath('%s/Android/Sdk' % environ_cp['APPDATA'])
  elif is_macos():
    default_sdk_path = '%s/library/Android/Sdk' % environ_cp['HOME']
  else:
    default_sdk_path = '%s/Android/Sdk' % environ_cp['HOME']

  def valid_sdk_path(path):
    return (os.path.exists(path) and
            os.path.exists(os.path.join(path, 'platforms')) and
            os.path.exists(os.path.join(path, 'build-tools')))

  android_sdk_home_path = prompt_loop_or_load_from_env(
      environ_cp,
      var_name='ANDROID_SDK_HOME',
      var_default=default_sdk_path,
      ask_for_var='Please specify the home path of the Android SDK to use.',
      check_success=valid_sdk_path,
      error_msg=('Either %s does not exist, or it does not contain the '
                 'subdirectories "platforms" and "build-tools".'))

  platforms = os.path.join(android_sdk_home_path, 'platforms')
  api_levels = sorted(os.listdir(platforms))
  api_levels = [x.replace('android-', '') for x in api_levels]

  def valid_api_level(api_level):
    return os.path.exists(
        os.path.join(android_sdk_home_path, 'platforms',
                     'android-' + api_level))

  android_api_level = prompt_loop_or_load_from_env(
      environ_cp,
      var_name='ANDROID_API_LEVEL',
      var_default=api_levels[-1],
      ask_for_var=('Please specify the Android SDK API level to use. '
                   '[Available levels: %s]') % api_levels,
      check_success=valid_api_level,
      error_msg='Android-%s is not present in the SDK path.')

  build_tools = os.path.join(android_sdk_home_path, 'build-tools')
  versions = sorted(os.listdir(build_tools))

  def valid_build_tools(version):
    return os.path.exists(
        os.path.join(android_sdk_home_path, 'build-tools', version))

  android_build_tools_version = prompt_loop_or_load_from_env(
      environ_cp,
      var_name='ANDROID_BUILD_TOOLS_VERSION',
      var_default=versions[-1],
      ask_for_var=('Please specify an Android build tools version to use. '
                   '[Available versions: %s]') % versions,
      check_success=valid_build_tools,
      error_msg=('The selected SDK does not have build-tools version %s '
                 'available.'))

  write_action_env_to_bazelrc('ANDROID_BUILD_TOOLS_VERSION',
                              android_build_tools_version, save_path)
  write_action_env_to_bazelrc('ANDROID_SDK_API_LEVEL', android_api_level,
                              save_path)
  write_action_env_to_bazelrc('ANDROID_SDK_HOME', android_sdk_home_path,
                              save_path)


def get_ndk_api_level(environ_cp, android_ndk_home_path):
  """Gets the appropriate NDK API level to use for the provided Android NDK path."""

  # First check to see if we're using a blessed version of the NDK.
  properties_path = '%s/source.properties' % android_ndk_home_path
  if is_windows():
    properties_path = cygpath(properties_path)
  with open(properties_path, 'r') as f:
    filedata = f.read()

  revision = re.search(r'Pkg.Revision = (\d+)', filedata)
  if revision:
    ndk_version = revision.group(1)
  else:
    raise Exception('Unable to parse NDK revision.')
  if int(ndk_version) not in _SUPPORTED_ANDROID_NDK_VERSIONS:
    print('WARNING: The NDK version in %s is %s, which is not '
          'supported by Bazel (officially supported versions: %s). Please use '
          'another version. Compiling Android targets may result in confusing '
          'errors.\n' %
          (android_ndk_home_path, ndk_version, _SUPPORTED_ANDROID_NDK_VERSIONS))

  # Now grab the NDK API level to use. Note that this is different from the
  # SDK API level, as the NDK API level is effectively the *min* target SDK
  # version.
  platforms = os.path.join(android_ndk_home_path, 'platforms')
  api_levels = sorted(os.listdir(platforms))
  api_levels = [
      x.replace('android-', '') for x in api_levels if 'android-' in x
  ]

  def valid_api_level(api_level):
    return os.path.exists(
        os.path.join(android_ndk_home_path, 'platforms',
                     'android-' + api_level))

  android_ndk_api_level = prompt_loop_or_load_from_env(
      environ_cp,
      var_name='ANDROID_NDK_API_LEVEL',
      var_default='21',  # 21 is required for ARM64 support.
      ask_for_var=('Please specify the (min) Android NDK API level to use. '
                   '[Available levels: %s]') % api_levels,
      check_success=valid_api_level,
      error_msg='Android-%s is not present in the NDK path.')

  return android_ndk_api_level


def create_android_build_rules(save_path):
  # Android configs. Bazel needs to have --cpu and --fat_apk_cpu both set to the
  # target CPU to build transient dependencies correctly. See
  # https://docs.bazel.build/versions/master/user-manual.html#flag--fat_apk_cpu
  write_to_bazelrc("build:android --cxxopt=-std=c++14", save_path)
  write_to_bazelrc("build:android --host_cxxopt=-std=c++14", save_path)
  write_to_bazelrc("build:android --define iree_is_android=true", save_path)
  write_to_bazelrc("build:android --define=with_xla_support=false", save_path)
  write_to_bazelrc("build:android --crosstool_top=//external:android/crosstool",
                   save_path)
  write_to_bazelrc(
      "build:android --host_crosstool_top=@bazel_tools//tools/cpp:toolchain",
      save_path)
  write_to_bazelrc("build:android_arm --config=android", save_path)
  write_to_bazelrc("build:android_arm --cpu=armeabi-v7a", save_path)
  write_to_bazelrc("build:android_arm --fat_apk_cpu=armeabi-v7a", save_path)
  write_to_bazelrc("build:android_arm64 --config=android", save_path)
  write_to_bazelrc("build:android_arm64 --cpu=arm64-v8a", save_path)
  write_to_bazelrc("build:android_arm64 --fat_apk_cpu=arm64-v8a", save_path)
  write_to_bazelrc("build:android --noenable_platform_specific_config",
                   save_path)
  write_to_bazelrc("build:android --copt=-w", save_path)


def main():
  if len(sys.argv) > 1:
    local_bazelrc = sys.argv[1]
  else:
    local_bazelrc = os.path.join(os.path.dirname(__file__),
                                 "configured.bazelrc")
  with open(local_bazelrc, "wt") as bazelrc:
    write_platform(bazelrc)

  # Configuring ANDROID environment variables.
  input = get_input("Would you like to build for Android (y or n)? ")
  if input.strip() == "y":
    environ_cp = dict(os.environ)
    create_android_ndk_rule(environ_cp, local_bazelrc)
    create_android_sdk_rule(environ_cp, local_bazelrc)
    create_android_build_rules(local_bazelrc)


if __name__ == '__main__':
  main()

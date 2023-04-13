# Copyright 2023 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""configure script to get build parameters from user."""

from importlib.machinery import SourceFileLoader
import inspect
import json
import os
import shlex
import subprocess
import sys

# pylint: disable=g-import-not-at-top
try:
  from shutil import which
except ImportError:
  from distutils.spawn import find_executable as which
# pylint: enable=g-import-not-at-top

CURRENT_DIR = os.path.dirname(
    os.path.abspath(inspect.getfile(inspect.currentframe())))

DOCKER_IMAGE_SHORTNAME_DICT = {
    "base": {
        "latest": "base-bleeding-edge",
        "stable": "base"
    },
    "nvidia": {
        "latest": "nvidia-bleeding-edge",
        "stable": "nvidia"
    }
}


def run_shell(cmd, allow_non_zero=False, stderr=None):
  cmd = shlex.split(cmd)
  if stderr is None:
    stderr = sys.stdout
  if allow_non_zero:
    try:
      output = subprocess.check_output(cmd, stderr=stderr)
    except subprocess.CalledProcessError as e:
      output = e.output
  else:
    output = subprocess.check_output(cmd, stderr=stderr)
  return output.decode('UTF-8').strip()


def is_nvidia_gpu_available():
  """This function verifies NVIDIA Docker runtime is installed and 
  available. It also checks NVIDIA GPU are available and the driver
  is installed."""

  docker_executable = which('docker')
  if docker_executable is None:
    raise RuntimeError("Docker is not installed or in the user path.")

  command_str = f"{docker_executable} info --format '{{{{json .}}}}'"
  data = json.loads(run_shell(command_str))

  if "nvidia" not in data["Runtimes"].keys():
    return False, "NVIDIA Docker Runtime is not available"

  nvidia_smi_executable = which('nvidia-smi')
  if nvidia_smi_executable is None:
    return False, "NVIDIA Driver is not installed or not in the user path."

  command_str = f"{nvidia_smi_executable} --query-gpu=gpu_name --format=csv"

  if run_shell(command_str).splitlines()[1:]:  # Skip header
    return True, None
  else:
    return False, "No NVIDIA GPU was found on this system."


def get_input(question, default_answer='', accepted_answers=None):
  accepted_answers = [x.strip().lower() for x in accepted_answers]

  while True:

    try:
      answer = input(f"{question} ")  # pylint: disable=bad-builtin
    except EOFError:
      answer = default_answer
    answer = (answer or default_answer).strip().lower()

    if answer not in accepted_answers:
      print(
          f"Unsupported answer received: {answer}. Expected: {accepted_answers}"
      )
      continue

    break

  return answer.strip().lower()


if __name__ == "__main__":

  if os.path.isfile(os.path.join(CURRENT_DIR, "docker-compose.yml")):
    if get_input(
        "A `docker-compose.yml` already exists. Are you certain you want to overwrite it [y/N]?",
        default_answer="n",
        accepted_answers=["y", "n"]) == "n":
      sys.exit(1)

  with open(os.path.join(CURRENT_DIR, "docker-compose.base.yml")) as f:
    docker_config = json.load(f)

  use_official_img = get_input(
      "Do you wish to use the official prebuild development containers [Y/n]?",
      default_answer="y",
      accepted_answers=["y", "n"]) == "y"

  use_bleeding = get_input(
      "Do you wish to use the bleeding-edge container (might be unstable) [Y/n]?",
      default_answer="y",
      accepted_answers=["y", "n"]) == "y"

  use_nvidia_gpu = False
  nvidia_gpu_available, err_msg = is_nvidia_gpu_available()
  if nvidia_gpu_available:
    use_nvidia_gpu = get_input("Do you wish to use NVIDIA GPU [y/N]?",
                               default_answer="n",
                               accepted_answers=["y", "n"]) == "y"
  else:
    print(f"NVIDIA GPUs are not available for use: {err_msg}")

  docker_img_shortname = DOCKER_IMAGE_SHORTNAME_DICT[
      "nvidia" if use_nvidia_gpu else "base"][
          "latest" if use_bleeding else "stable"]

  if use_official_img:
    del docker_config["services"]["iree-dev"]["build"]

    docker_iree_registry = SourceFileLoader(
        "docker_iree_registry",
        os.path.join(CURRENT_DIR,
                     "../build_tools/docker/get_image_name.py")).load_module()

    image_name = docker_iree_registry.find_image_by_name(docker_img_shortname)
    docker_config["services"]["iree-dev"]["image"] = image_name

  else:
    del docker_config["services"]["iree-dev"]["image"]
    dockerfile = f"build_tools/docker/dockerfiles/{docker_img_shortname}.Dockerfile"
    docker_config["services"]["iree-dev"]["build"]["dockerfile"] = dockerfile
    if (not os.path.isfile(os.path.join(CURRENT_DIR, "..", dockerfile))):
      raise FileNotFoundError(f"The file `{dockerfile}` does not exist.")

  if not use_nvidia_gpu:
    del docker_config["services"]["iree-dev"]["deploy"]

  with open(os.path.join(CURRENT_DIR, "docker-compose.yml"), "w") as f:
    json.dump(docker_config, f, indent=2)

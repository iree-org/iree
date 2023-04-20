#!/usr/bin/env python3

# Copyright 2023 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Configure script to setup VSCode DevContainer Environment."""

import json
import os
import pathlib
import shlex
import subprocess
import sys

from importlib.machinery import SourceFileLoader
from shutil import which

CURRENT_DIR = pathlib.Path(__file__).resolve().parent

DOCKER_IMAGE_SHORTNAME_DICT = {
    "base": "base-bleeding-edge",
    "nvidia": "nvidia-bleeding-edge",
}


def run_shell(cmd):
  cmd = shlex.split(cmd)
  return subprocess.run(cmd, check=True, text=True,
                        stdout=subprocess.PIPE).stdout


def is_nvidia_gpu_available():
  """This function verifies NVIDIA Docker runtime is installed and
  available. It also checks NVIDIA GPU are available and the driver
  is installed."""

  command_str = f"'{which('docker')}' info --format '{{{{json .}}}}'"
  data = json.loads(run_shell(command_str))

  if "nvidia" not in data["Runtimes"].keys():
    return (
        False, "NVIDIA Docker Runtime is not available. Please see: "
        "https://developer.nvidia.com/nvidia-container-runtime for additional "
        "information.")

  nvidia_smi_executable = which('nvidia-smi')
  if nvidia_smi_executable is None:
    return False, "NVIDIA Driver is not installed or not in the user path."

  command_str = f"{nvidia_smi_executable} --query-gpu=gpu_name --format=csv"

  if run_shell(command_str).splitlines()[1:]:  # Skip header
    return True, None
  else:
    return False, "No NVIDIA GPU was found on this system."


def _get_input(question, default_answer):
  try:
    answer = input(f"\n- {question} ")
  except EOFError:
    answer = default_answer

  return (answer or default_answer).strip().lower()


def get_input(question, default_answer='', accepted_answers=None):
  if accepted_answers is None:
    raise RuntimeError("Argument `accepted_answers` is None.")

  accepted_answers = [x.strip().lower() for x in accepted_answers]

  while True:
    answer = _get_input(question, default_answer)
    if answer not in accepted_answers:
      print(f"\tERROR: Unsupported answer received: {answer}."
            f"Expected: {accepted_answers}")
      continue
    break

  return answer


def get_input_path(question):

  while True:
    answer = _get_input(question, default_answer="")
    if answer != "" and not os.path.isdir(answer):
      print(f"\tERROR: Received path does not exist: `{answer}`")
      continue
    break

  return answer


if __name__ == "__main__":

  # ------------------------------------------------------------------------- #
  #                         Environment Verifications                         #
  # ------------------------------------------------------------------------- #

  docker_executable = which('docker')
  if docker_executable is None:
    raise RuntimeError(
        "Docker is not installed or in the user path. Please refer to: "
        "https://docs.docker.com/desktop/")

  try:
    run_shell(f"'{docker_executable}' compose version")
  except subprocess.CalledProcessError as e:
    raise RuntimeError(
        "Docker Compose must be installed in order to use IREE VS Code "
        "Development Container. Please refer to: "
        "https://docs.docker.com/compose/") from e

  docker_image_key = "base"

  # ------------------------------------------------------------------------- #
  #                               Mandatory Steps                             #
  # ------------------------------------------------------------------------- #

  # STEP 1: Verify the user doesn't have a pre-existing `docker-compose.yml`.
  #         If yes, confirm they want to overwrite it.
  if os.path.isfile(CURRENT_DIR / "docker-compose.yml"):
    if get_input(
        "A `docker-compose.yml` already exists. Are you certain you want to overwrite it [y/N]?",
        default_answer="n",
        accepted_answers=["y", "n"]) == "n":
      sys.exit(1)

  # STEP 2: Read the template configuration file
  with open(CURRENT_DIR / "docker-compose.base.yml") as f:
    docker_config = json.load(f)

  # STEP 3: Prebuilt vs Locally Built Containers
  use_official_img = get_input(
      "Do you wish to use the official prebuild development containers [Y/n]?",
      default_answer="y",
      accepted_answers=["y", "n"]) == "y"

  # ------------------------------------------------------------------------- #
  #                 Optional Mounting Points - Build & Cache                  #
  # ------------------------------------------------------------------------- #

  # STEP 4 [OPTIONAL]: Does the user want to mount a directory for CCACHE ?
  ccache_mount_cache = get_input_path(
      "[OPTIONAL] Enter the the directory path to mount as CCache Cache.\n"
      "  It will be mounted in the container under: `/home/iree/.cache/ccache` [Default: None]:"
  )
  if ccache_mount_cache:
    docker_config["services"]["iree-dev"]["volumes"].append(
        f"{ccache_mount_cache}:/home/iree/.cache/ccache:cached")

  # STEP 5 [OPTIONAL]: Does the user want to mount a directory for BUILD ?
  build_mount_cache = get_input_path(
      "[OPTIONAL] Enter the the directory path to mount as Build Directory.\n"
      "  It will mounted in the container under: `/home/iree/build` [Default: None]:"
  )
  if build_mount_cache:
    docker_config["services"]["iree-dev"]["volumes"].append(
        f"{build_mount_cache}:/home/iree/build:cached")

  # ------------------------------------------------------------------------- #
  #                Optional Deep Learning Accelerator Support                 #
  # ------------------------------------------------------------------------- #

  # STEP 6 [OPTIONAL]: Does the user want to use NVIDIA GPUs ?
  nvidia_gpu_available, err_msg = is_nvidia_gpu_available()

  if nvidia_gpu_available:
    if get_input("[OPTIONAL] Do you wish to use NVIDIA GPU [y/N]?",
                 default_answer="n",
                 accepted_answers=["y", "n"]) == "y":
      docker_image_key = "nvidia"

  else:
    print(f"[INFO] NVIDIA GPUs are not available for use: {err_msg}")

  if docker_image_key != "nvidia":
    del docker_config["services"]["iree-dev"]["deploy"]

  # ------------------------------------------------------------------------- #
  #            Setting the right docker image / container to build            #
  # ------------------------------------------------------------------------- #

  docker_img_shortname = DOCKER_IMAGE_SHORTNAME_DICT[docker_image_key]

  if use_official_img:
    del docker_config["services"]["iree-dev"]["build"]

    docker_iree_registry = SourceFileLoader(
        "docker_iree_registry",
        str(CURRENT_DIR /
            "../build_tools/docker/get_image_name.py")).load_module()

    image_name = docker_iree_registry.find_image_by_name(docker_img_shortname)
    docker_config["services"]["iree-dev"]["image"] = image_name

  else:
    del docker_config["services"]["iree-dev"]["image"]

    dockerfile = f"build_tools/docker/dockerfiles/{docker_img_shortname}.Dockerfile"
    docker_config["services"]["iree-dev"]["build"]["dockerfile"] = dockerfile

    if (not os.path.isfile(CURRENT_DIR / ".." / dockerfile)):
      raise FileNotFoundError(f"The file `{dockerfile}` does not exist.")

  docker_compose_filepath = os.path.join(CURRENT_DIR, "docker-compose.yml")
  with open(docker_compose_filepath, "w") as f:
    json.dump(docker_config, f, indent=2)

  # ------------------------------------------------------------------------- #
  #                                  SUCCESS                                  #
  # ------------------------------------------------------------------------- #

  print("\n" + "=" * 80)
  print(f"\nSuccess! Wrote Docker Compose file to `{docker_compose_filepath}`.")

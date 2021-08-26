#!/bin/bash
# Copyright 2021 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

set -euo pipefail

./scripts/git/submodule_versions.py init

# TODO: Pull the docker configuration into a separate file as with Kokoro

# Make the source repository available and launch containers in that
# directory.
DOCKER_RUN_ARGS=(
  --volume="${PWD}:${PWD}"
  --workdir="${PWD}"
)

# Delete the container after the run is complete.
DOCKER_RUN_ARGS+=(--rm)


# Run as the current user and group. If only it were this simple...
DOCKER_RUN_ARGS+=(--user="$(id -u):$(id -g)")


# The Docker container doesn't know about the users and groups of the host
# system. We have to tell it. This is just a mapping of IDs to names though.
# The thing that really matters is the IDs, so the key thing is that Docker
# writes files as the same ID as the current user, which we set above, but
# without the group and passwd file, lots of things get upset because they
# don't recognize the current user ID (e.g. `whoami` fails). Bazel in
# particular looks for a home directory and is not happy when it can't find
# one.
# So we make the container share the host mapping, which guarantees that the
# current user is mapped. If there was any user or group in the container
# that we cared about, this wouldn't necessarily work because the host and
# container don't necessarily map the ID to the same user. Luckily though,
# we don't.
# We don't just mount the real /etc/passwd and /etc/group because Google
# Linux workstations do some interesting stuff with user/group permissions
# such that they don't contain the information about normal users and we
# want these scripts to be runnable locally for debugging.
# Instead we dump the results of `getent` to some fake files.
fake_etc_dir="/tmp/fake_etc"
mkdir -p "${fake_etc_dir}"

fake_group="${fake_etc_dir}/group"
fake_passwd="${fake_etc_dir}/passwd"

getent group > "${fake_group}"
getent passwd > "${fake_passwd}"

DOCKER_RUN_ARGS+=(
  --volume="${fake_group}:/etc/group:ro"
  --volume="${fake_passwd}:/etc/passwd:ro"
)


# Bazel stores its cache in the user home directory by default. It's
# possible to override this, but that would require changing our Bazel
# startup options, which means polluting all our scripts and making them not
# runnable locally. Instead, we give it a special home directory to write
# into. We don't just mount the user home directory (or some subset thereof)
# for two reasons:
#   1. We probably don't want Docker to just write into the user's home
#      directory when running locally.
#   2. BUILDKITE_LOCAL_SSD (configured by agent environment hook) points to a
#      directory mounted on a local SSD whereas the home directory is on the
#      persistent boot disk. It turns out that makes a huge difference in
#      performance for Bazel running with execution (not with RBE) because it is
#      IO bound with many cores.
fake_home_dir="${BUILDKITE_LOCAL_SSD}/docker_home_dir"
mkdir -p "${fake_home_dir}"
DOCKER_RUN_ARGS+=(
  --volume="${fake_home_dir}:${fake_home_dir}"
  -e "HOME=${fake_home_dir}"
)

# Add a ramdisk and use the bazelrc to set --sandbox_base to it. This
# dramatically improves performance with a large number of cores. See
# https://github.com/bazelbuild/bazel/issues/11868 and
# https://docs.bazel.build/versions/main/command-line-reference.html#flag--sandbox_base
# We do this with a bazelrc because it's an environment-specific setting and
# this keeps the inside_docker.sh script somewhat environment-agnostic (someone
# on a similar linux box or already in a container can just run it directly)

SYSTEM_BAZELRC="${fake_etc_dir}/bazel.bazelrc"
echo "build --sandbox_base=/dev/shm" > "${SYSTEM_BAZELRC}"
DOCKER_RUN_ARGS+=(
  --tmpfs=/dev/shm
  --volume="${SYSTEM_BAZELRC}:/etc/bazel.bazelrc:ro"
)

echo "Beginning run inside Docker container"
set -x
docker run "${DOCKER_RUN_ARGS[@]}" \
  gcr.io/iree-oss/cmake-bazel-frontends-swiftshader@sha256:103676490242311b9fad841294689a7ce1c755b935a21d8d898c25cfe3ec15e8 \
  build_tools/buildkite/bazel/tf_integrations/inside_docker.sh

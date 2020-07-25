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

# Functions for setting up Docker containers to run on Kokoro

# Sets up files and environment to enable running all our Kokoro docker scripts.
# In particular, does some shenanigans to enable running with the current user.
# Some of this setup is only strictly necessary for Bazel, but it doesn't hurt
# for anything else.
# Requires that KOKORO_ROOT and KOKORO_ARTIFACTS_DIR have been set
# Sets the environment variable DOCKER_RUN_ARGS to be used by subsequent
# `docker run` invocations.
function docker_setup() {
    # Setup to run docker as the current user.
    # Provide a place to mount files that would normally be under /etc/
    # We don't just mount the real /etc/passwd and /etc/group because Google
    # Linux workstations do some interesting stuff with user/group permissions
    # such that they don't contain the information about normal users and we
    # want these scripts to be runnable locally for debugging.
    local fake_etc_dir="${KOKORO_ROOT?}/fake_etc"
    # Bazel wants a home directory
    # TODO: more explanation if this works
    mkdir -p "${fake_etc_dir?}"

    local fake_group="${fake_etc_dir?}/group"
    local fake_passwd="${fake_etc_dir?}/passwd"

    getent group > "${fake_group?}"
    getent passwd > "${fake_passwd?}"

    local fake_home="${KOKORO_ROOT?}/fake_home"
    mkdir -p "${fake_home}"

    local workdir="${KOKORO_ARTIFACTS_DIR?}/github/iree"

    DOCKER_RUN_ARGS=(
      # Run as the current user and group
      --user="$(id -u):$(id -g)"
      # Make the source repository available
      --volume="${workdir?}:${workdir?}"
      --workdir="${workdir?}"
      # Tell docker about the host users and groups. Bazel needs this
      # information, but it also makes some other things more pleasant.
      --volume="${fake_group?}:/etc/group:ro"
      --volume="${fake_passwd?}:/etc/passwd:ro"
      --volume="${fake_home?}:${HOME?}"
      # Make gcloud credentials available. This isn't necessary when running
      # in GCE but enables using this script locally with RBE.
      --volume="${HOME?}/.config/gcloud:${HOME?}/.config/gcloud:ro"
      # Delete the container after
      --rm
    )
}

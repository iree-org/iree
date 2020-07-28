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
    # Make the source repository available and launch containers in that
    # directory.
    local workdir="${KOKORO_ARTIFACTS_DIR?}/github/iree"
    DOCKER_RUN_ARGS=(
      --volume="${workdir?}:${workdir?}"
      --workdir="${workdir?}"
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
    local fake_etc_dir="${KOKORO_ROOT?}/fake_etc"
    mkdir -p "${fake_etc_dir?}"

    local fake_group="${fake_etc_dir?}/group"
    local fake_passwd="${fake_etc_dir?}/passwd"

    getent group > "${fake_group?}"
    getent passwd > "${fake_passwd?}"

    DOCKER_RUN_ARGS+=(
      --volume="${fake_group?}:/etc/group:ro"
      --volume="${fake_passwd?}:/etc/passwd:ro"
    )


    # Bazel stores its cache in the user home directory by default. It's
    # possible to override this, but that would require changing our Bazel
    # startup options, which means polluting all our scripts and making them not
    # runnable locally. Instead, we give it a special home directory to write
    # into. We don't just mount the user home directory (or some subset thereof)
    # for two reasons:
    #   1. We probably don't want Docker to just write into the user's home
    #      directory when running locally.
    #   2. When running with Kokoro, we mount a local scratch SSD to KOKORO_ROOT
    #      whereas the home directory is on the persistent SSD boot disk. It
    #      turns out that makes a huge difference in performance for Bazel
    #      running with local execution (not with RBE) because it is IO bound at
    #      64 cores.
    local fake_home_dir="${KOKORO_ROOT?}/fake_home"
    mkdir -p "${fake_home_dir}"

    DOCKER_RUN_ARGS+=(
      --volume="${fake_home_dir?}:${HOME?}"
    )

    # Make gcloud credentials available. This isn't necessary when running in
    # GCE but enables using this script locally with RBE.
    DOCKER_RUN_ARGS+=(
      --volume="${HOME?}/.config/gcloud:${HOME?}/.config/gcloud:ro"
    )
}

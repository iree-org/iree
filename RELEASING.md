# IREE Releasing

This file documents the extant release process that IREE uses. This process
and the automation (such as it is) has grown over many years and is due for
a refresh. However, in the interests of documenting what exists, we attempt
to do so here.

## Nightly Core Releases

IREE development is primarily driven via automated nightly release snapshots.
These are scheduled automatically each day by the
`schedule_candidate_release.yml` workflow, which selects a green commit from
main (for non optional CI tasks), created a tag of the format
`iree-{X.Y.ZrcYYYYMMDD}` and schedules automation to populate the release.

The `build_package.yml` workflow then runs jobs to do builds for all
platforms and packages, finally triggering the
`validate_and_publish_release.yml` workflow.

Release artifacts are uploaded as a GitHub
[pre release](https://github.com/iree-org/iree/releases) and an index of files
is updated by periodic automation at https://iree.dev/pip-release-links.html.

Some debugging notes for this process are available here:
https://iree.dev/developers/debugging/releases/.

### Nightly Release Packages

A number of packages are produced automatically:

* `iree-dist-*.tar.xz` (manylinux x86_64 and aarch64): Install image of the
  binaries and development assets needed to use or depend on the C/C++ parts
  of the project.
* `iree-base-compiler`: Binary Python wheels
* `iree-base-runtime`: Binary Python wheels
* `iree-tools-tf` and `iree-tools-tflite`: Pure Python wheels

#### Linux Builds

Binary Linux packages are built using a custom `manylinux` based Docker image
hosted here:
https://github.com/iree-org/base-docker-images/pkgs/container/manylinux_x86_64
using isolated self-hosted runners (only used for building checked in code) of
sufficient size for building large components and GitHub managed runners for
smaller components. The project aims to target all non-EOL Python versions with
Linux builds on x86_64 and aarch64.

#### Windows Builds

Windows builds are built using GitHub-hosted runners. Due to the cost, the
project aims to target the most recent version of Python only while building
version N-1 for the first year of the lifecycle of the next version.

Only the Python `iree-base-compiler` and `iree-base-runtime` packages are
built for Windows.

The release is published even if the Windows build fails. When this happens, it
is fixed forward for the next snapshot.

#### MacOS Builds

MacOS builds are performed using GitHub-hosted runners. Due to the cost, the
project aims to target the most recent version of Python only while building
version N-1 for the first year of the lifecycle of the next version.

Only the Python `iree-base-compiler` and `iree-base-runtime` packages are
built for MacOS.

The release is published even if the MacOS build fails. When this happens, it
is fixed forward for the next snapshot.

## Retention

The project will keep pre-release tagged releases on its releases page for a
minimum of 6 months. Releases older than this can be purged.

## Distribution to Package Registries

The following package registry projects are managed as part of the IREE
release process:

### PyPI

* https://pypi.org/project/iree-base-compiler/
* https://pypi.org/project/iree-base-runtime/
* https://pypi.org/project/iree-turbine/
* https://pypi.org/project/iree-tools-tf/
* https://pypi.org/project/iree-tools-tflite/

Deprecated projects no longer updated:

* https://pypi.org/project/iree-compiler/ (replaced by iree-base-compiler)
* https://pypi.org/project/iree-runtime/ (replaced by iree-base-runtime)
* https://pypi.org/project/iree-runtime-instrumented/ (functionality is
  included in the main iree-runtime package)
* https://pypi.org/project/iree-tools-xla/ (functionality is no longer needed)


## Build Promotion

There are presently two build promotion processes documented:

* Releasing IREE core packages:
  https://iree.dev/developers/general/release-management/
* Releasing iree-turbine packages:
  https://github.com/iree-org/iree-turbine/blob/main/docs/releasing.md

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
`candidate-{YYYYMMDD}.{BUILDNUM}` and schedules automation to populate the
release.

The `build_package.yml` workflow then runs jobs to do builds for all
platforms and packages, finally triggering the 
`validate_and_publish_release.yml` workflow.

Release artifacts are uploaded as a GitHub
[pre release](https://github.com/iree-org/iree/releases) and an index of files
is updated by periodic automation at https://iree.dev/pip-release-links.html.

Some debugging notes for this process are available here: 
https://iree.dev/developers/debugging/releases/

### Nightly Release Packages

A number of packages are produced automatically:

* `iree-dist-*.tar.xz` (manylinux x86_64 and aarch64): Install image of the
  binaries and development assets needed to use or depend on the C/C++ parts
  of the project.
* `iree-compiler`: Binary Python wheels
* `iree-runtime`: Binary Python wheels
* `iree-tools-tf` and `iree-tools-tflite`: Pure Python wheels

#### Linux Builds

Binary Linux packages are built using a custom `manylinux` based Docker 
image hosted here: https://github.com/nod-ai/base-docker-images/pkgs/container/manylinux_x86_64
(TODO: this repository of Docker images should be moved into `iree-org`) using
isolated self-hosted runners (only used for building checked in code) of
sufficient size for building large components and GitHub managed runners for
smaller components. The project aims to target all non-EOL Python versions with
Linux builds on x86_64 and aarch64.

#### Windows Builds

Windows builds are built using GitHub managed large Windows runners. Due to the
cost, the project aims to target the most recent version of Python only while
building version N-1 for the first year of the lifecycle of the next version.

Only the Python `iree-compiler` and `iree-runtime` packages are built for
Windows.

The release is published even if the MacOS build fails. When this happens, it
is fixed forward for the next snapshot.

#### MacOS Builds

MacOS builds are performed using self hosted MacOS runners in a dedicated
post-submit pool. Due to the cost, the project aims to target the most recent 
version of Python only while building version N-1 for the first year of the 
lifecycle of the next version.

Only the Python `iree-compiler` and `iree-runtime` packages are built for
MacOS.

The release is published even if the MacOS build fails. When this happens, it
is fixed forward for the next snapshot.

## Build Promotion

There are presently two build promotion processes documented:

* Old one focused purely on releasing IREE core packages: 
https://iree.dev/developers/general/release-management/
* New one driven by the Torch frontend: 
https://github.com/nod-ai/SHARK-Turbine/blob/main/docs/releasing.md

The versioning scheme for `iree-turbine` (which is 
[in the process of being added to IREE](https://groups.google.com/g/iree-discuss/c/Bk58qwhaPEU)) is rooted on the then-current PyTorch released version, with
optional date-based dev/pre-release suffixes (i.e. `rcYYYYMMDD` or 
`devYYYYMMDD`) or intra PyTorch releases (i.e. `postVVVV`). This process is
being trialed to correspond with the 2.3.0 release of PyTorch. In this scenario,
the pinned nightly build of IREE is considered current and promoted as part of
the Turbine release to PyPI (and the release is marked as not pre-release on the
GitHub releases page).

Promotions are done roughly monthly or at need. The schedule is shifted to
account for extra factors as needed.

In the future, we would like to adopt a real versioning scheme (beyond the
nightly calver+build number scheme) and manage promotion and pinning of the
core IREE dep more explicitly and in alignment with how downstreams are using
it.

## Retention

The project will keep pre-release tagged releases on its releases page for a
minimum of 6 months. Releases older than this can be purged.

## Distribution to Package Registries

The following package registry projects are managed as part of the IREE
release process:

### PyPI

* https://pypi.org/project/iree-compiler/
* https://pypi.org/project/iree-runtime/
* https://pypi.org/project/iree-turbine/
* https://pypi.org/project/shark-turbine/ (transitional until switched to
  iree-turbine)
* https://pypi.org/project/iree-tools-tf/
* https://pypi.org/project/iree-tools-tflite/

Deprecated projects no longer updated:

* https://pypi.org/project/iree-runtime-instrumented/ (functionality is
  included in the main iree-runtime package)
* https://pypi.org/project/iree-tools-xla/ (functionality is no longer needed)

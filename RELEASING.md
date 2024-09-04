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
https://iree.dev/developers/debugging/releases/.

### Nightly Release Packages

A number of packages are produced automatically:

* `iree-dist-*.tar.xz` (manylinux x86_64 and aarch64): Install image of the
  binaries and development assets needed to use or depend on the C/C++ parts
  of the project.
* `iree-compiler`: Binary Python wheels
* `iree-runtime`: Binary Python wheels
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

Only the Python `iree-compiler` and `iree-runtime` packages are built for
Windows.

The release is published even if the Windows build fails. When this happens, it
is fixed forward for the next snapshot.

#### MacOS Builds

MacOS builds are performed using GitHub-hosted runners. Due to the cost, the
project aims to target the most recent version of Python only while building
version N-1 for the first year of the lifecycle of the next version.

Only the Python `iree-compiler` and `iree-runtime` packages are built for
MacOS.

The release is published even if the MacOS build fails. When this happens, it
is fixed forward for the next snapshot.

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


## Build Promotion

There are presently two build promotion processes documented:

* Old one focused purely on releasing IREE core packages:
https://iree.dev/developers/general/release-management/
* New one driven by the Torch frontend and documented below.

The versioning scheme for
[iree-turbine](https://github.com/iree-org/iree-turbine) is rooted on the
then-current PyTorch released version, with optional date-based dev/pre-release
suffixes (i.e. `rcYYYYMMDD` or `devYYYYMMDD`) or intra PyTorch releases
(i.e. `postVVVV`).

This process is being trialed to correspond with the 2.3.0 release of PyTorch.
In this scenario, the pinned nightly build of IREE is considered current and
promoted as part of the Turbine release to PyPI (and the release is marked as
not pre-release on the GitHub releases page).

Promotions are done roughly monthly or at need. The schedule is shifted to
account for extra factors as needed.

In the future, we would like to adopt a real versioning scheme (beyond the
nightly calver+build number scheme) and manage promotion and pinning of the
core IREE dep more explicitly and in alignment with how downstreams are using
it.

### Steps to Promote

There are multiple release artifacts that are deployed from this project:

* shark-turbine wheel (transitional while switching to iree-turbine)
* iree-turbine wheel
* iree-compiler wheels
* iree-runtime wheels

Typically we deploy IREE compiler and runtime wheels along with a turbine
release, effectively promoting a nightly.

#### Building Artifacts

Start with a clean clone of iree-turbine:

```
cd scratch
git clone git@github.com:iree-org/iree-turbine.git
cd iree-turbine
```

Build a pre-release:

```
./build_tools/build_release.py --core-version 2.3.0 --core-pre-version=rcYYYYMMDD
```

Build an official release:

```
./build_tools/build_release.py --core-version 2.3.0
```

This will download all deps, including wheels for all supported platforms and
Python versions for iree-compiler and iree-runtime. All wheels will be placed
in the `wheelhouse/` directory.


#### Testing

TODO: Write a script for this.

```
python -m venv wheelhouse/test.venv
source wheelhouse/test.venv/bin/activate
pip install -f wheelhouse iree-turbine[testing]
# Temp: tests require torchvision.
pip install -f wheelhouse torchvision
pytest core/tests
```

#### Push

From the testing venv, verify that everything is sane:

```
pip freeze
```

Push IREE deps (if needed/updated):

```
twine upload wheelhouse/iree_compiler-* wheelhouse/iree_runtime-*
```

Push built wheels:

```
twine upload wheelhouse/iree_turbine-* wheelhouse/shark_turbine-*
```

#### Install from PyPI and Sanity Check

TODO: Script this

From the testing venv:

```
pip uninstall -y shark-turbine iree-turbine iree-compiler iree-runtime
pip install iree-turbine
pytest core/tests
```

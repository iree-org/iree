# Python Deployment

These scripts assist with building Python packages and pushing them to
[PyPI (the Python Package Index)](https://pypi.org/). See also

* The Python Packaging User Guide: <https://packaging.python.org/en/latest/>
* Our release management documentation:
  https://iree.dev/developers/general/release-management/

## Overview

See comments in scripts for canonical usage. This page includes additional
notes.

### Package building

These scripts build all packages we maintain, for all Python versions and
platforms that we support:

* [`build_linux_packages.sh`](./build_linux_packages.sh)
* [`build_macos_packages.sh`](./build_macos_packages.sh)
* [`build_windows_packages.ps1`](./build_windows_packages.ps1)

To assist with environment setup, we use a
[manylinux Docker image](https://github.com/iree-org/base-docker-images/blob/main/dockerfiles/manylinux_x86_64.Dockerfile)
for Linux builds and these scripts on other platforms:

* [`install_macos_deps.sh`](./install_macos_deps.sh)
* [`install_windows_deps.ps1`](./install_windows_deps.ps1)

### Version management

These scripts handle versioning across packages, including considerations like
major, minor, and patch levels (`X.Y.Z`), as well as suffixes like
`rc20241107` or `dev+{git hash}`:

* [`compute_common_version.py`](./compute_common_version.py)
* [`compute_local_version.py`](./compute_local_version.py)
* [`promote_whl_from_rc_to_final.py`](./promote_whl_from_rc_to_final.py)

### PyPI deployment

These scripts handle promoting nightly releases packages to stable and pushing
to PyPI:

* [`promote_whl_from_rc_to_final.py`](./promote_whl_from_rc_to_final.py)
* [`pypi_deploy.sh`](./pypi_deploy.sh)

Both of these scripts expect to have the dependencies from
[`pypi_deploy_requirements.txt`](./pypi_deploy_requirements.txt) installed.
This can be easily managed by using a Python virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r ./pypi_deploy_requirements.txt
```

### Release index publication

The [`generate_release_index.py`](./generate_release_index.py) script,
run as part of
[`.github/workflows/publish_website.yml`](../../.github/workflows/publish_website.yml),
scrapes release artifact URLs from https://github.com/iree-org/iree/releases
(and the release pages for other ecosystem projects) to generate the release
index published at https://iree.dev/pip-release-links.html.

The release index can be used like so:

```bash
python -m pip install \
  --pre --find-links https://iree.dev/pip-release-links.html \
  iree-base-compiler iree-base-runtime
```

## Debugging manylinux builds

We build releases under a manylinux derived docker image. When all goes well,
things are great, but when they fail, it often implicates something that has
to do with Linux age-based arcana. In this situation, just getting to the
shell and building/poking can be the most useful way to triage.

Here is the procedure:

```
[host ~/iree]$ docker run --interactive --tty --rm -v $(pwd):/work/iree gcr.io/iree-oss/manylinux2014_x86_64-release:prod
[root@c8f6d0041d79 /]# export PATH=/opt/python/cp310-cp310/bin:$PATH
[root@c8f6d0041d79 /]# python --version
Python 3.10.4


# Two paths for further work.
# Option A: Build like a normal dev setup (i.e. if allergic to Python
# packaging and to triage issues that do not implicate that).
[root@c8f6d0041d79 ]# cd /work/iree
[root@c8f6d0041d79 iree]# pip install wheel cmake ninja pybind11 numpy
[root@c8f6d0041d79 iree]# cmake -GNinja -B ../iree-build/ -S . -DCMAKE_BUILD_TYPE=Release -DIREE_BUILD_PYTHON_BINDINGS=ON
[root@c8f6d0041d79 iree]# cd ../iree-build/
[root@c8f6d0041d79 iree-build]# ninja

# Options B: Creates Python packages exactly as the CI/scripts do.
# (to be used when there is Python arcana involved). The result is organized
# differently from a usual dev flow and may be subsequently more confusing to
# the uninitiated.
[root@c8f6d0041d79 iree]# pip wheel compiler/
[root@c8f6d0041d79 iree]# pip wheel runtime/
```

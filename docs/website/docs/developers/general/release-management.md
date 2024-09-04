---
icon: octicons/package-16
---

# Release management

IREE cuts automated releases via a workflow that is
[triggered daily](https://github.com/iree-org/iree/blob/main/.github/workflows/schedule_candidate_release.yml).
The only constraint placed on the commit that is released is that it has
[passed all required CI checks](https://github.com/iree-org/iree/blob/main/build_tools/scripts/get_latest_green.sh).
These are published on GitHub with the "pre-release" status. For debugging this
process, see the [Release debugging playbook](../debugging/releases.md).

We periodically promote one of these candidates to a "stable" release by
removing the "pre-release" status. This makes it show up as a "latest" release
on GitHub. We also push the Python packages for this release to PyPI.

## Release status

| Package | Release status |
| -- | -- |
GitHub release (stable) | [![GitHub Release](https://img.shields.io/github/v/release/iree-org/iree)](https://github.com/iree-org/iree/releases/latest)
GitHub release (nightly) | [![GitHub Release](https://img.shields.io/github/v/release/iree-org/iree?include_prereleases)](https://github.com/iree-org/iree/releases)
Python iree-compiler | [![PyPI version](https://badge.fury.io/py/iree-compiler.svg)](https://badge.fury.io/py/iree-compiler)
Python iree-runtime | [![PyPI version](https://badge.fury.io/py/iree-runtime.svg)](https://badge.fury.io/py/iree-runtime)

## Running a release

A pinned issue tracking the next release should be filed like
<https://github.com/iree-org/iree/issues/18380>. Developers authoring patches
that include major or breaking changes should coordinate merge timing and
contribute release notes on those issues.

### Picking a candidate to promote

After approximately one month since the previous release, a new release should
be promoted from nightly release candidates.

When selecting a candidate we use the following criteria:

1. ⪆4 days old so that problems with it may have been spotted
2. Contains no P0 regressions vs the previous stable release
3. LLVM submodule commit ideally exists upstream (no cherry picks or patches)
4. Includes packages for all platforms, including macOS and Windows

When you've identified a potential candidate, comment on the tracking issue with
the proposal and solicit feedback. People may point out known regressions or
request that some feature make the cut.

### Promoting a candidate to stable

1. (Authorized users only) Push to PyPI using
    [pypi_deploy.sh](https://github.com/iree-org/iree/blob/main//build_tools/python_deploy/pypi_deploy.sh)

    * For Googlers, the password is stored at <http://go/iree-pypi-password>

2. Open the release on GitHub. Rename the release from "candidate" to "stable",
    uncheck the option for "pre-release", and check the option for "latest".

    ![rename_release](./release-renaming.png)

    ![promote_release](./release-promotion.png)

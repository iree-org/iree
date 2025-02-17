---
icon: octicons/versions-16
---

# Versioning scheme

A shared version format is used for the packages

* `iree-base-compiler` (formally named `iree-compiler`)
* `iree-base-runtime` (formally named `iree-runtime`)
* `iree-turbine`

## Overview

Type of build | Version format | Version example
------------- | -------------- | ---------------
Stable release (PyPI) | `X.Y.Z` | `3.0.0`
Nightly release (GitHub `schedule`) | `X.Y.ZrcYYYYMMDD` | `3.0.0rc20241029`
Dev release (GitHub `pull_request`) | `X.Y.Z.devNN` | `3.0.0.dev+6d55a11`
Local build | `X.Y.Z.devNN` | `3.0.0.dev+6d55a11`

### Key

Identifier | Explanation
---------- | -----------
`X` | Major version
`Y` | Minor version
`Z` | Patch version
`rc` | release candidate (`main` branch)
`dev` | developer build (code on pull request branches)
`YYYY` | Year, e.g. `2024`
`MM` | Month, e.g. `10`
`DD` | Day, e.g. `29`
`NN` | git commit hash, e.g. `6d55a11`

## Composition of version numbers

A release number is in the format of `X.Y.Z` (MAJOR.MAJOR.PATCH)

* `X` and `Y` are defined as shared version numbers between all packages.
* The patch level `Z` MAY be incremented individually.
* A PATCH release contains only bug fixes and the version `Z` (`x.y.Z`) MUST be
  incremented. A bug fix is an internal change that fixes incorrect behavior
  and MUST NOT introduce breaking changes.
* A MINOR release (unlike SemVer) as well as a MAJOR release MAY contain
  backwards-incompatible, breaking changes, like API changes and removals and
  furthermore bug fixes and new features.

### Development and nightly releases

* Development builds (e.g. from a regular CI) MUST be released with a version
  number defined as `X.Y.Z.dev+NN`, where `NN` is the git commit hash.
* Nightly releases MUST be released with a version number defined as `X.Y.ZrcYYYYMMDD`.
* The intent is to promote a recent, high quality release candidate to a final
  version.

Binary stamps and tools will continue to report the original release candidate version.

## Semantics

The following semantics apply:

* If the version `X` (`X.y.z`) is increased for one package, the version number
  change MUST be adopted by all (other) packages. The same applies for the
  version `Y` (`x.Y.Z`).
* If the version `X` or `Y` are changed, `Z` MUST be set `0`.
* After a regular (non-patch) release, `Y` MUST be increased to ensure
  precedence of nightly builds.
  For example:
    * The latest stable release published on November 15th 2024 is versioned as
      version `3.0.0`.
    * The next nightly builds are released as `3.1.0rc20241116`.
    * The next stable release is released as `3.1.0` or `4.0.0`.

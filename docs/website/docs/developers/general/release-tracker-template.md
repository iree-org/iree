## Overview

> [!IMPORTANT]
> The X.YY.Z release is scheduled for YYYY-MM-DD.

Previous stable release: [vA.BB.C](https://github.com/iree-org/iree/releases/tag/vA.BB.C).
The process follows https://iree.dev/developers/general/release-management/.

## Release checklist

- [ ] Watch for major/breaking changes and decide to either batch them with this release or defer them until the next release
- [ ] Choose release candidates from nightly releases to promote. The candidates should contain no major regressions and should include all packages, even those marked `experimental` in [`.github/workflows/build_package.yml`](https://github.com/iree-org/iree/blob/main/.github/workflows/build_package.yml) like macOS and Windows packages.
  - Chosen: [`iree-X.YY.ZrcYYYYMMDD`](https://github.com/iree-org/iree/releases/tag/iree-X.YY.ZrcYYYYMMDD) at https://github.com/iree-org/iree/commit/COMMIT (YYYY-MM-DD). It has packages for all platforms, including macOS and Windows, and no major regressions vs vA.BB.C.
- [ ] Compile release notes
  - Changes since the previous release: https://github.com/iree-org/iree/compare/vA.BB.C...iree-X.YY.ZrcYYYYMMDD
- [ ] Push IREE packages to PyPI
- [ ] Create a new release on GitHub
- [ ] Push iree-turbine packages to PyPI and create a new release on GitHub
- [ ] Increment version.json files
  - `X.YY.Z.dev` -> `X.(YY+1).0.dev`
- [ ] Close this issue and open a new one for the next release

## Testing instructions

(Update this once release candidates are selected. Versions should match within each repository.)

```
pip install \
  --find-links https://iree.dev/pip-release-links.html \
  iree-base-compiler==X.YY.ZrcYYYYMMDD \
  iree-base-runtime==X.YY.ZrcYYYYMMDD \
  iree-tools-tf==YYYYMMDD.NNNN \
  iree-tools-tflite==YYYYMMDD.NNNN
```

## Release notes

---
---

# IREE vX.YY.Z Release Notes

**Release Candidate:** `iree-X.YY.ZrcYYYYMMDD`
**PRs:** k since vA.BB.C
**VMFB Bytecode Version:** N.M ([un]changed from vA.BB.C)
**HAL Module Version:** O.P ([un]changed from vA.BB.C)


## Breaking changes
* **ITEM** DESCRIPTION https://github.com/iree-org/iree/pull/NNNNN

## Notable changes

### Compiler

### Runtime

### Developer tools

## New contributors

## Full changelog
**Commit history**: https://github.com/iree-org/iree/compare/vA.BB.C...vX.YY.Z

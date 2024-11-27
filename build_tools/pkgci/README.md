# PkgCI Scripts

This directory contains scripts and configuration for "PkgCI", which
is based on building packages and then flowing those to followon jobs.

The prior/traditional CI attempted to do all steps as various kinds of source
builds at head vs a split package/test style of workflow. It can mostly
be found in the `cmake` directory but is also scattered around.

This directory generally corresponds to "pkgci_" prefixed workflows. Over
time, as this CI flow takes over more of the CI pipeline, the traditional
CI will be reduced to outlier jobs and policy checks.

### Development notes

Testing venv setup using packages:

```bash
python3.11 ./setup_venv.py /tmp/.venv --fetch-git-ref=5b0740c97a33ed

# Activate the venvs and test it
source /tmp/.venv/bin/activate
iree-compile --version
# IREE (https://iree.dev):
#   IREE compiler version 3.1.0.dev+5b0740c97a33edce29e753b14b9ff04789afcc53 @ 5b0740c97a33edce29e753b14b9ff04789afcc53
#   LLVM version 20.0.0git
#   Optimized build
```

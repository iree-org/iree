# PkgCI Scripts

This directory contains scripts and configuration for the "new" CI, which
is based on building packages and then flowing those to followon jobs.

The traditional CI attempted to do all steps as various kinds of source
builds at head vs a split package/test style of workflow. It can mostly
be found in the `cmake` directory but is also scattered around.

This directory generally corresponds to "pkgci_" prefixed workflows. Over
time, as this CI flow takes over more of the CI pipeline, the traditional
CI will be reduced to outlier jobs and policy checks.

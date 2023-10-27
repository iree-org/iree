# Releasing

IREE cuts automated releases via a workflow that is
[triggered daily](https://github.com/openxla/iree/blob/main/.github/workflows/schedule_candidate_release.yml).
The only constraint placed on the commit that is released is that it has passed
all CI checks. These are published on GitHub with the "pre-release" status. For
debugging this process, see the
[Debugging releases playbook](/docs/developers/debugging/releases.md).

We periodically promote one of these candidates to a "stable" release by
removing the "pre-release" status. This makes it show up as a "latest" release
on GitHub. We also push the Python packages for this release to PyPI.

The criteria for selecting this candidate is a bit more involved.

## Picking a Candidate to Promote

When selecting a candidate we use the following criteria:

1. ⪆4 days old so that problems with it may have been spotted.
2. Contains no major regressions vs the previous stable release.
3. (Ideally) LLVM submodule commit exists upstream (no cherry picks or patches)

When you've identified a potential candidate, email the iree-discuss list with
the proposal and solicit feedback. People may point out known regressions or
request that some feature make the cut.

## Releasing

1. (Googlers only) Push to PyPI using
   [pypi_deploy.sh](/build_tools/python_deploy/pypi_deploy.sh) and the
   password stored at <http://go/iree-pypi-password>.

2. Open the release on GitHub. Rename the release from “candidate" to “stable",
   uncheck the option for “pre-release”, and check the option for "latest".

   ![rename_release](/docs/developers/assets/rename_release.png)

   ![promote_release](/docs/developers/assets/promote_release.png)

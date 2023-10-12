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

## Coupling to the Google Integrate

The Google team that manages these stable releases at the moment is also
responsible for integrating the IREE source code into Google's monorepo. For
convenience, we select a candidate pre-release, attempt to integrate it into
Google's monorepo and then promote it to stable if that was successful without
cherry picks that would affect the quality of the release (because they wouldn't
be present in the promoted version).

This gives some additional validation to the release because it is stress-tested
running in a different environment and we not-infrequently catch some issues. We
do not currently have a way to add cherry-picks to a release, so if cherry-picks
for functional issues are required, we skip promoting the candidate to "stable".

This coupling introduces some additional constraints to the process that are not
inherent. It would be perfectly fine to start promoting candidates based on some
other process, but since the same people are managing both, we've coupled these
so we don't have to keep track of as many different versions.

As the project matures, we will likely remove this coupling. In particular we
will likely start integrating into Google's monorepo at a faster cadence than
the stable releases, so a 1:1 mapping there will not make sense.

The PyPI password is also currently stored in Google's internal secret
management system, so for others to manage the deployment, we would need to
store it elsewhere with appropriate ACLs.

At the point where others want to engage in the release process, we can easily
remove any coupling to any Google processes.

## Picking a Candidate to Promote

When selecting a candidate we use the following criteria:

1. ⪆4 days old so that problems with it may have been spotted.
2. Contains no P0 regressions vs the previous stable release.
3. LLVM submodule commit exists upstream (no cherry picks or patches) and
   matches a commit already integrated into Google's monorepo

The constraint on LLVM version is largely due to our current process for doing
so. We aim to lift this limitation and if the process were decoupled from the
Google integration (see
[Coupling to the Google Integrate](#coupling-to-the-google-integrate)), it would
go away anyway.

There is currently no specific tracking for P0 regressions (process creation in
progress). When you've identified a potential candidate, email the iree-discuss
list with the proposal and solicit feedback. People may point out known
regressions or request that some feature make the cut.

## Releasing

1. (Googlers only) Integrate into Google's monorepo, following
   <http://go/iree-g3-integrate-playbook>. If OSS-relevant cherry-picks were
   required to complete this, STOP: do not promote the candidate.

2. (Googlers only) Push to PyPI using
   [pypi_deploy.sh](/build_tools/python_deploy/pypi_deploy.sh) and the
   password stored at <http://go/iree-pypi-password>.

3. Open the release on GitHub. Rename the release from “candidate" to “stable",
   uncheck the option for “pre-release”, and check the option for "latest".

   ![rename_release](/docs/developers/assets/rename_release.png)

   ![promote_release](/docs/developers/assets/promote_release.png)

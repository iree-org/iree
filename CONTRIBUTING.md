# How to Contribute

We'd love to accept your patches and contributions to this project. There are
just a few small guidelines you need to follow.

## Contributor License Agreement

Contributions to this project must be accompanied by a Contributor License
Agreement. You (or your employer) retain the copyright to your contribution;
this simply gives us permission to use and redistribute your contributions as
part of the project. Head over to <https://cla.developers.google.com/> to see
your current agreements on file or to sign a new one.

You generally only need to submit a CLA once, so if you've already submitted one
(even if it was for a different project), you probably don't need to do it
again.

## Changes Accepted

Please file issues before doing substantial work; this will ensure that others
don't duplicate the work and that there's a chance to discuss any design issues.

Changes only tweaking style are unlikely to be accepted unless they are applied
consistently across the project. Most of the code style is derived from the
[Google Style Guides](http://google.github.io/styleguide/) for the appropriate
language and is generally not something we accept changes on (as clang-format
and clang-tidy handle that for us). The compiler portion of the project follows
[MLIR style](https://mlir.llvm.org/getting_started/DeveloperGuide/#style-guide).
Improvements to code structure and clarity are welcome but please file issues to
track such work first.

## Code reviews

All submissions, including submissions by project members, require review. We
use GitHub pull requests (PRs) for this purpose. Consult
[GitHub Help](https://help.github.com/articles/about-pull-requests/) for more
information on using pull requests.

## Presubmits

Several of our presubmit builds will only run automatically if you are a project
collaborator. Otherwise a collaborator must label the PR with "kokoro:run". If
you are sending code changes to the project, please ask to be added as a
collaborator, so that these can run automatically. It is generally expected that
PRs will only be merged when all presubmit checks are passing. In some cases,
pre-existing failures may be ignored.

## Merging

After review and presubmit checks, PRs should be merged with a "squash and
merge". The squashed commit summary should match the PR title and the commit
description should match the PR body. Accordingly, please write these as you
would a helpful commit message. Please also keep PRs small (focused on a single
issue) to streamline review and ease later culprit-finding. It is assumed that
the PR author will merge their change unless they ask someone else to merge it
for them (e.g. because they don't have write access).

## Peculiarities

Our documentation on
[repository management](https://github.com/google/iree/blob/main/docs/repository_management.md)
has more information on some of the oddities in our repository setup and
workflows. For the most part, these should be transparent to normal developer
workflows.

## Community Guidelines

This project follows
[Google's Open Source Community Guidelines](https://opensource.google.com/conduct/).

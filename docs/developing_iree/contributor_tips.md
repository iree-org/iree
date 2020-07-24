# Contributor Tips

This is an opinionated guide documenting workflows that some members of the team
have found useful. It is focused on meta-tooling, not on IREE code specifically
(you will find the latter in the [Developer Overview](developer_overview.md)) It
is certainly possible to use workflows other than these, but some common tasks,
especially for maintainers will likely be made easier if you use these flows. It
assumes a basic knowledge of `git` and GitHub and suggests some specific ways of
using it.

## Git Structure

We tend to use the "triangular" or "forking" workflow. Develop primarily on a
clone of the repository on your development machine. Any local branches named
the same as persistent branches from the
[main repository](https://github.com/google/iree) (currently `main`, `google`,
and `stable`) are pristine (though potentially stale) copies. You only
fastforward these to match upstream and otherwise do development on other
branches. When sending PRs, you push to a different branch on your public fork
and create the PR from there.

### Setup

1.  Create a fork of the main repository.

2.  Create a local git repository with remotes `upstream` (the main repository)
    and `origin` (your personal fork). To list your current remotes `git remote
    -v`.

    a. If you already cloned from the main repository (e.g. by following the
    getting started guide):

    ```shell
    # From your existing git repo
    git remote rename origin upstream
    git add remote origin git@github.com:<github_username>/iree.git
    ```

    b. If you haven't already cloned:

    ```shell
    # From whatever directory under which you want to nest your repo
    git clone git@github.com:<github_username>/iree.git
    cd iree
    git remote add upstream git@github.com:google/iree.git
    ```

    This is especially important for maintainers who have write access (so can
    push directly to the main repository) and admins who have elevated
    privileges (so can push directly to protected branches). These names are
    just suggestions, but you might find some scripts where the defaults are for
    remotes named like this. For extra safety, you can make it difficult to push
    directly to upstream by setting the push url to something invalid: `git
    remote set-url --push upstream DISABLE`, which requires re-enabling the push
    URL explicitly before pushing.

3.  Use a script like
    [git_update.sh](https://github.com/google/iree/blob/main/scripts/git/git_update.sh)
    to easily synchronize `main` with `upstream`. Submodules make this is a
    little trickier than it should be. You can also add this as a git alias.

    ```shell
    git config alias.update "! /path/to/git-update"
    git config alias.sync "update main"
    ```

## Useful Tools

*   GitHub CLI (https://github.com/cli/cli). A CLI for interacting with GitHub.
    Most importantly, it allows scripting the creation of pull requests.
*   Refined GitHub Chrome and Firefox Extension:
    https://github.com/sindresorhus/refined-github. Nice extension that adds a
    bunch of features to the GitHub UI.
*   Squashed Merge Messages Chrome and Firefox Extension:
    https://github.com/zachwhaley/squashed-merge-message. Simple extension that
    implements the desired pattern for IREE squash-merge and merge-commit
    messages, copying the PR title and body.
*   VSCode: https://code.visualstudio.com/. The most commonly used IDE amongst
    IREE developers.

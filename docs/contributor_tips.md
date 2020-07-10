# Maintainer Tips

This is an opinionated guide documenting workflows that some members of the team
have found useful. It is focused on meta-tooling, not on IREE code specifically
(you will find the latter in the
[Developer Overview](../developer_overview.md)) It is certainly possible to use
workflows other than these, but some common tasks, especially for maintainers
will likely be made easier if you use these flows. It assumes a basic knowledge
of `git` and GitHub and suggests some specific ways of using it.

## Git Structure

Work primarily from a clone of the repository on your development machine. Any
local branches named the same as persistent branches from the
[main repository](https://github.com/google/iree) (currently `main`, `google`,
and `stable`) are pristine (though potentially stale) copies. You only
fastforward these to match upstream and otherwise do development on other
branches. When sending PRs, you push to a different branch on your fork and
create the PR from there.

### Setup

1. Create a fork of the main repository.
2. Clone from this repository or the main repository.
3. Rename your remotes to `upstream` and `fork`, so that it's abundantly clear
   which is which. This is especially important for maintainers who have write
   access (so can push directly to the main repository) and admins who have
   elevated privileges (so can push directly to protected branches). These names
   are just suggestions, but you might find some scripts that assume the remotes
   are named like this.

   TODO(gcmn): These are just the names I use. We could poll the team to
   establish a convention here for the scripts.

4. Create git aliases to easily synchronize your persistent branches with
   `upstream`.

    ```
    [alias]
        update = "!f() { \
          git checkout ${1?} && git pull upstream ${1?} --ff-only && git submodule update --init && git status; \
        }; f"
        sync = ! ([[ -z "$(git status --porcelain)" ]] && git update google && git update main) || (echo "Sync failed" && git status && false)
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

# Contributor Tips

This is an opinionated guide documenting workflows that some members of the team
have found useful. It is focused on meta-tooling, not on IREE code specifically
(you will find the latter in the
[Developer Overview](./developer_overview.md)). It is certainly possible to use
workflows other than these, but some common tasks, especially for maintainers
will likely be made easier if you use these flows. It assumes a basic knowledge
of `git` and GitHub and suggests some specific ways of using it.

## Git Structure

We tend to use the "triangular" or "forking" workflow. Develop primarily on a
clone of the repository on your development machine. Any local branches named
the same as persistent branches from the
[main repository](https://github.com/openxla/iree) (currently `main` and
`stable`) are pristine (though potentially stale) copies. You only fastforward
these to match upstream and otherwise do development on other branches. When
sending PRs, you push to a different branch on your public fork and create the
PR from there.

### Setup

1. Create a fork of the main repository.

2. Create a local git repository with remotes `upstream` (the main repository)
    and `origin` (your personal fork). To list your current remotes
    `git remote -v`.

    a. If you already cloned from the main repository (e.g. by following the
    getting started guide):

    ```shell
    # From your existing git repo
    $ git remote rename origin upstream
    $ git remote add origin https://github.com/<github_username>/iree.git
    ```

    b. If you haven't already cloned:

    ```shell
    # From whatever directory under which you want to nest your repo
    $ git clone https://github.com/<github_username>/iree.git
    $ cd iree
    $ git remote add upstream https://github.com/openxla/iree.git
    ```

    This is especially important for maintainers who have write access (so can
    push directly to the main repository) and admins who have elevated
    privileges (so can push directly to protected branches). These names are
    just suggestions, but you might find some scripts where the defaults are for
    remotes named like this. For extra safety, you can make it difficult to push
    directly to upstream by setting the push url to something invalid: `git
    remote set-url --push upstream DISABLE`, which requires re-enabling the push
    URL explicitly before pushing. You can wrap this behavior in a custom git
    command like
    [git-sudo](https://gist.github.com/GMNGeoffrey/42dd9a9792390094a43bdb69659320c0).

3. Use a script like
    [git_update.sh](/build_tools/scripts/git/git_update.sh)
    to easily synchronize `main` with `upstream`. Submodules make this is a
    little trickier than it should be. You can also turn this into a git command
    by adding it to your path as `git-update`.

#### Git Config

These are some additional options you could put in your top-level `.gitconfig`
or repository-specific `.git/config` files that are conducive the recommended
workflow

```ini
[push]
  default = current
[alias]
  # Delete branches that you pushed and have been deleted upstream, e.g. because
  # the PR was merged.
  gone = ! "git fetch -p  && git for-each-ref --format '%(refname:short) %(upstream:track)' | awk '$2 == \"[gone]\" {print $1}' | xargs -r git branch -D"
  # Update from upstream (custom command) and delete obsolete local branches.
  sync = ! (git update main && git gone)
  # Create a new branch based off of main (requires a clean working directory).
  new = "!f(){ \\\ngit checkout main && git switch -c $1; \\\n}; f"
  # Display branches in a useful "latest last" format
  br = for-each-ref --sort=committerdate refs/heads/ --format='%(HEAD) %(color:yellow)%(refname:short)%(color:reset) - %(color:red)%(objectname:short)%(color:reset) - %(contents:subject) (%(color:green)%(committerdate:relative)%(color:reset))'
  # `git git foo` -> `git foo` typo fixer
  git = "!f(){ \\\n git \"$@\"; \\\n}; f"
  # Get the git root directory
  root = rev-parse --show-toplevel
  # checkout, but also sync submodules
  ch = "!f() { \\\n git checkout \"$@\"; git submodule sync && git submodule update --init; \\\n}; f"
  # See the diff for a PR branch vs the main branch
  diffmain = diff --merge-base main
  # See only the files that differ vs the main branch
  whatsout = diffmain --name-only
[checkout]
  # If the checkout command
  defaultRemote = origin
[pull]
  # When pulling, only complete the pull if its a clean fast forward.
  ff = only
[remote]
  # Push to your fork (origin) by default
  pushDefault = origin
[url "ssh://git@github.com/"]
  # Pull with https (so no auth required), but push with ssh.
  pushInsteadOf = https://github.com/
```

## Useful Tools

* GitHub CLI (<https://github.com/cli/cli>). A CLI for interacting with GitHub.
    Most importantly, it allows scripting the creation of pull requests.
* Refined GitHub Chrome and Firefox Extension:
    <https://github.com/sindresorhus/refined-github>. Nice extension that adds a
    bunch of features to the GitHub UI.
* VSCode: <https://code.visualstudio.com/>. The most commonly used IDE amongst
    IREE developers.
* [ccache](./ccache.md)

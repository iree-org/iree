#!/usr/bin/env python
"""Exports a submodule head to a branch in one of our fork repos.

If patches have been applied to a local submodule for which we
host a fork repository, this script will "export" those patches
to specially maintained branches in the fork such that the
head commit is available indefinitely and can be used safely
as a submodule commit.

This will create branches in the fork named like:
  sm-iree-{branch}

(i.e. all historical patch commits for `main` will be on
sm-iree-main or if they derive from a specific integrate or
branch on sm-iree-integrate_....)

The forks have branch protection enabled for branches starting
with "sm-" so that they cannot be force pushed.

This is all accomplished by creating special merge commits and
associating trees with them explicitly. Most folks don't need
to understand that: they just need to know that if you run
this script, the current commit at your submodule head will
be available "forever".

Usage:
  ./scripts/export_submodule_head.py third_party/llvm-project
  ./scripts/export_submodule_head.py third_party/stablehlo
  ./scripts/export_submodule_head.py third_party/torch-mlir
"""
from typing import Optional

import argparse
from pathlib import Path
import re
import shlex
import subprocess
import sys


def export_submodule_head(args, submodule_rel_path: str):
    super_repo_path = args.repo_path
    super_repo_name = args.repo_path.name
    super_branch = git_current_branch(repo_dir=args.repo_path)
    print(f"Super-repo '{super_repo_name}' is on branch '{super_branch}'")
    submodule_path = super_repo_path / submodule_rel_path
    print(f"Operating on submodule {submodule_path}")
    check_origin_update_help(submodule_path)
    git_fetch(repository="origin", repo_dir=submodule_path)
    submodule_head, submodule_summary = git_current_commit(repo_dir=submodule_path)
    print(f"Submodule at {submodule_head}\n  {submodule_summary}")
    submodule_merge_base = git_merge_base(
        submodule_head, "origin/main", repo_dir=submodule_path
    )
    if submodule_merge_base == submodule_head:
        print("Submodule commit is upstream. Nothing to do.")
        return 0

    submodule_branch = args.submodule_branch or f"sm-{super_repo_name}-{super_branch}"
    print(
        f"Submodule merge base {submodule_merge_base} diverges from upstream. Will persist on {submodule_branch}."
    )

    # Get the remote topic head.
    remote_topic_head = git_remote_head(
        "origin", f"refs/heads/{submodule_branch}", repo_dir=submodule_path
    )

    # Early exit if precisely at this commit.
    if remote_topic_head == submodule_head:
        print(f"Submodule branch {submodule_branch} is already at {submodule_head}")
        return 0

    # If the branch does not exist, just push to it and exit.
    if not remote_topic_head:
        print(f"Submodule branch {submodule_branch} does not exist. Pushing.")
        git_exec(
            ["push", "origin", f"{submodule_head}:refs/heads/{submodule_branch}"],
            repo_dir=submodule_path,
        )
        print("PLEASE IGNORE ANY NOTICE ABOUT CREATING A PR")
        return 0

    # Check if the submodule_head is an ancestor of the current remote_topic_head
    # and exit if so (it is already reachable).
    try:
        git_exec(
            ["merge-base", "--is-ancestor", submodule_head, remote_topic_head],
            repo_dir=submodule_path,
        )
        print(
            f"Commit {submodule_head} is reachable from remote branch {submodule_branch}. Doing nothing."
        )
        return
    except subprocess.CalledProcessError as e:
        # If not an ancestor, returncode will be 1. On general error, it will be
        # something else.
        if e.returncode != 1:
            raise

    # Create a splice commit that is based on the tree of the current submodule head
    # and has parents of the current submodule head and the remote topic head.
    # Note that the current branch is not touched, the commit is just created in the
    # ether. We can push it to the remote topic branch to complete the splice.
    print(f"Submodule head {submodule_head} is not on {submodule_branch}. Splicing.")
    splice_commit = git_exec(
        [
            "commit-tree",
            submodule_head + "^{tree}",
            "-p",
            submodule_head,
            "-p",
            remote_topic_head,
            "-m",
            f"Splice submodule rebase {submodule_head} onto {remote_topic_head}",
        ],
        repo_dir=submodule_path,
        capture_output=True,
    ).strip()
    print(f"Created splice commit {splice_commit}: pushing")
    git_exec(
        ["push", "origin", f"{splice_commit}:refs/heads/{submodule_branch}"],
        repo_dir=submodule_path,
    )


def git_current_commit(*, repo_dir=None) -> tuple[str, str]:
    output = git_exec(
        ["log", "-n", "1", "--pretty=format:%H %s (%an on %ci)"],
        capture_output=True,
        repo_dir=repo_dir,
        quiet=True,
    )
    output = output.strip()
    parts = output.split(" ")
    # Return commit, full_summary
    return parts[0], output


def git_current_branch(*, repo_dir=None):
    return git_exec(
        ["rev-parse", "--abbrev-ref", "HEAD"],
        repo_dir=repo_dir,
        quiet=True,
        capture_output=True,
    ).strip()


def check_origin_update_help(repo_dir):
    existing_url = git_exec(
        ["remote", "get-url", "--push", "origin"],
        capture_output=True,
        repo_dir=repo_dir,
        quiet=True,
    )
    existing_url = existing_url.strip()
    if existing_url.startswith("https://github.com/"):
        new_url = existing_url.replace("https://github.com/", "git@github.com:", 1)
        print(
            "Your push URL is for GitHub HTTPS. Just in case if you are only set up "
            "to push with SSH, here is a one-liner to update it:"
        )
        print(f"  (cd {repo_dir} && git remote set-url --push origin {new_url})")
        return False
    return True


def git_fetch(*, repository=None, ref=None, repo_dir=None):
    args = ["fetch"]
    if repository:
        args.append(repository)
    if ref is not None:
        args.append(ref)
    git_exec(args, repo_dir=repo_dir)


def git_merge_base(ref1, ref2, *, repo_dir=None) -> str:
    return git_exec(
        ["merge-base", ref1, ref2], quiet=True, capture_output=True, repo_dir=repo_dir
    ).strip()


def git_remote_head(remote: str, head: str, repo_dir=None) -> Optional[str]:
    # Get the remote head (i.e. "refs/heads/main") commit or None.
    args = ["ls-remote", "--heads", remote, head]
    output = git_exec(args, capture_output=True, repo_dir=repo_dir)
    lines = output.strip().splitlines(keepends=False)
    if not lines:
        return None

    def extract_commit(line):
        parts = re.split("\\s+", line)
        commit = parts[0]
        return commit

    return next(extract_commit(l) for l in lines)


def git_exec(args, *, repo_dir, quiet=False, capture_output=False):
    full_args = ["git"] + args
    full_args_quoted = [shlex.quote(a) for a in full_args]
    if not quiet:
        print(f"  ++ EXEC: (cd {repo_dir} && {' '.join(full_args_quoted)})")
    if capture_output:
        return subprocess.check_output(full_args, cwd=repo_dir).decode("utf-8")
    else:
        subprocess.check_call(full_args, cwd=repo_dir)


def main(args):
    return export_submodule_head(args, args.submodule)


def parse_arguments(argv):
    repo_root = Path(__file__).resolve().parent.parent.parent
    parser = argparse.ArgumentParser(description="Submodule exporter")
    parser.add_argument("--repo-path", default=repo_root, type=Path)
    parser.add_argument("--submodule-branch")
    parser.add_argument("submodule")
    args = parser.parse_args(argv)
    return args


if __name__ == "__main__":
    main(parse_arguments(sys.argv[1:]))

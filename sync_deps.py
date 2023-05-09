#!/usr/bin/env python
### AUTO-GENERATED: DO NOT EDIT
### Casual developers and CI bots invoke this to do the most
### efficient checkout of dependencies.
### Cross-repo project development should use the 
### 'openxla-workspace' dev tool for more full featured setup.
### Update with: openxla-workspace pin

PINNED_VERSIONS = {
  "iree": "f3e70439739249dd8f24c6930e8b04666f8a95d6",
  "xla": "dad64948516e3672b3e2518945831a70b5e90b81",
  "jax": "68ba54241c3fe6a192c26e8cd8f49690111bbaa3"
}

ORIGINS = {
  "iree": "https://github.com/openxla/iree.git",
  "jax": "https://github.com/google/jax.git",
  "xla": "https://github.com/openxla/xla.git"
}

SUBMODULES = {
  "iree": 1
}



### Update support:

import argparse
from pathlib import Path
import re
import shlex
import subprocess


def main():
  parser = argparse.ArgumentParser(description="Source deps sync")
  parser.add_argument(
      "--exclude-submodule",
      nargs="*",
      help="Exclude submodules by regex (relative to '{project}:{path})")
  parser.add_argument("--exclude-dep",
                      nargs="*",
                      help="Excludes dependencies by regex")
  parser.add_argument("--depth",
                      type=int,
                      default=0,
                      help="Fetch revisions with --depth")
  parser.add_argument("--submodules-depth",
                      type=int,
                      default=0,
                      help="Update submodules with --depth")
  args = parser.parse_args()

  workspace_dir = Path(__file__).resolve().parent.parent
  for repo_name, revision in PINNED_VERSIONS.items():
    # Exclude this dep?
    exclude_repo = False
    for exclude_pattern in (args.exclude_dep or ()):
      if re.search(exclude_pattern, repo_name):
        exclude_repo = True
    if exclude_repo:
      print(f"Excluding {repo_name} based on --exclude-dep")
      continue

    print(f"Syncing {repo_name}")
    repo_dir = workspace_dir / repo_name
    if not repo_dir.exists():
      # Shallow clone
      print(f"  Cloning {repo_name}...")
      repo_dir.mkdir()
      run(["init"], repo_dir)
      run(["remote", "add", "origin", ORIGINS[repo_name]], repo_dir)
    # Checkout detached head.
    fetch_args = ["fetch"]
    if args.depth > 0:
      fetch_args.extend(["--depth=1"])
    fetch_args.extend(["origin", revision])
    run(fetch_args, repo_dir)
    run(["-c", "advice.detachedHead=false", "checkout", revision], repo_dir)
    if SUBMODULES.get(repo_name):
      print(f"  Initializing submodules for {repo_name}")
      cp = run(["submodule", "status"],
               repo_dir,
               silent=True,
               capture_output=True)
      submodules = []
      for submodule_status_line in cp.stdout.decode().splitlines():
        submodule_status_parts = submodule_status_line.split()
        submodule_path = submodule_status_parts[1]
        exclude_submodule = False
        for exclude_pattern in (args.exclude_submodule or ()):
          if re.search(exclude_pattern, f"{repo_name}:{submodule_path}"):
            exclude_submodule = True
        if exclude_submodule:
          print(f"  Excluding {submodule_path} based on --exclude-submodule")
          continue
        submodules.append(submodule_path)

      update_args = ["submodule", "update", "--init"]
      if args.submodules_depth > 0:
        update_args.extend(["--depth", "1"])
      update_args.extend(["--"])
      update_args.extend(submodules)
      run(update_args, repo_dir)


def run(args,
        cwd,
        *,
        capture_output: bool = False,
        check: bool = True,
        silent: bool = False):
  args = ["git"] + args
  args_text = ' '.join([shlex.quote(arg) for arg in args])
  if not silent:
    print(f"  [{cwd}]$ {args_text}")
  cp = subprocess.run(args, cwd=str(cwd), capture_output=capture_output)
  if check and cp.returncode != 0:
    addl_info = f":\n({cp.stderr.decode()})" if capture_output else ""
    raise RuntimeError(f"Git command failed: {args_text} (from {cwd})"
                       f"{addl_info}")
  return cp


if __name__ == "__main__":
  main()

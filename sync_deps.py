#!/usr/bin/env python
### AUTO-GENERATED: DO NOT EDIT
### Casual developers and CI bots invoke this to do the most
### efficient checkout of dependencies.
### Cross-repo project development should use the 
### 'openxla-workspace' dev tool for more full featured setup.
### Update with: openxla-workspace pin

PINNED_VERSIONS = {
  "iree": "5c005b7de092ca48aee76356b77f09141cdc5599",
  "xla": "179d4899a1fbe5d8a8dc462edd90c504a9421686",
  "jax": "fb46d3d084c2de13ceb0344c4adc241df910cb4d"
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
    run(["fetch", "--depth=1", "origin", revision], repo_dir)
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

      run([
          "submodule", "update", "--init", "--depth", "1",
          "--recommend-shallow", "--"
      ] + submodules, repo_dir)


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

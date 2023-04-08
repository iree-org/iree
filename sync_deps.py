#!/usr/bin/env python
### AUTO-GENERATED: DO NOT EDIT
### Casual developers and CI bots invoke this to do the most
### efficient checkout of dependencies.
### Cross-repo project development should use the 
### 'openxla-workspace' dev tool for more full featured setup.
### Update with: openxla-workspace pin

# iree: 7faa624fb 2023-04-07 10:54:01 -0700 : Turn the functionality in iree_setup_toolchain.cmake into a macro. (#12955)
# jax: 891b5b60c 2023-04-07 12:05:26 -0700 : Merge pull request #15458 from jakevdp:fix-debug-exports
# xla: 66966c7e3 2023-04-07 11:13:43 -0700 : Fix Apple M1 CPU freq detection.

PINNED_VERSIONS = {
  "iree": "7faa624fb733fdb368bfb41c3469cdd7a3e3f67f",
  "xla": "66966c7e3c243029ebccc91b3d81aa006df5a645",
  "jax": "891b5b60c8dfc1973a74e376841135d1d97b73e1"
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

from pathlib import Path
import shlex
import subprocess

def main():
  workspace_dir = Path(__file__).resolve().parent.parent
  for repo_name, revision in PINNED_VERSIONS.items():
    repo_dir = workspace_dir / repo_name
    if not repo_dir.exists():
      # Shallow clone
      print(f"Cloning {repo_name}...")
      repo_dir.mkdir()
      run(["init"], repo_dir)
      run(["remote", "add", "origin", ORIGINS[repo_name]], repo_dir)
    # Checkout detached head.
    run(["fetch", "--depth=1", "origin", revision], repo_dir)
    run(["-c", "advice.detachedHead=false", "checkout", revision], repo_dir)
    if SUBMODULES.get(repo_name):
      print(f"Initializing submodules for {repo_name}")
      run(["submodule", "update", "--init", "--depth", "1",
           "--recommend-shallow"], repo_dir)


def run(args,
        cwd,
        *,
        capture_output: bool = False,
        check: bool = True,
        silent: bool = False):
  args = ["git"] + args
  args_text = ' '.join([shlex.quote(arg) for arg in args])
  if not silent:
    print(f"[{cwd}]$ {args_text}")
  cp = subprocess.run(args, cwd=str(cwd), capture_output=capture_output)
  if check and cp.returncode != 0:
    addl_info = f":\n({cp.stderr.decode()})" if capture_output else ""
    raise RuntimeError(f"Git command failed: {args_text} (from {cwd})"
                       f"{addl_info}")
  return cp


if __name__ == "__main__":
  main()

#!/usr/bin/env python
### AUTO-GENERATED: DO NOT EDIT
### Casual developers and CI bots invoke this to do the most
### efficient checkout of dependencies.
### Cross-repo project development should use the 
### 'openxla-workspace' dev tool for more full featured setup.
### Update with: openxla-workspace pin

PINNED_VERSIONS = {
  "iree": "e95f6f37fb38366cd27111b671b642abee1a37c1",
  "xla": "8bbdc65c78cdde9ac3fdfce9307ada50b9114706",
  "jax": "c42aae9fd74615aeac89b87b357fc60431bce99a"
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

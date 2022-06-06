# Buildkite Repository Hooks

Repository hooks to be run during the job lifecycle of every Buildkite job.
These are symlinked from `.buildkite/hooks` because the hooks are required to be
at that path by Buildkite
([feature request to make this customizable](https://forum.buildkite.community/t/custom-repository-hook-location/2081)). These hooks are *sourced*, so must be
bash.

# Rules for .claude/ Directory

**Every token costs.** Anything checked into `.claude/` is paid for on every invocation by every developer. Guard this directory jealously.

## CLAUDE.md

**Bar: Benefits ALL developers on EVERY use.**

We do not yet have project-wide instructions that meet this bar. Keep your instructions in:
- `~/.claude/CLAUDE.md` (personal global)
- `CLAUDE.local.md` (personal per-project, gitignored)

`/init` is quite decent at generating a `CLAUDE.md` and you can write your own prompt based on the kind of work you do to guide it. There's some [good example init-like prompts](https://github.com/kaushikgopal/dotfiles/blob/master/.ai/commands/initialize.md) out there to use as reference.

Bad example: [pytorch/CLAUDE.md](https://github.com/pytorch/pytorch/blob/2dd529df0092799f68ee7afcf52338276906706a/CLAUDE.md) - includes testing syntax that benefits a fraction of uses.

## Skills

**Bar: Benefits nearly all developers. Durable (useful 1 year from now). Graduated from well-honed workflows.**

Skills trigger automatically and their descriptions appear in context regardless of whether they are used (context bloat). A skill is ready for inclusion in the main repository when you've explained the same approach to Claude multiple times across many sessions and it has stabilized into a reliable workflow _and_ that workflow is one adopted by several other contributors. Otherwise, keep it local or in shared plugins.

**Requirements:**
- Broadly applicable (not for a single pass, single refactoring, or single API)
- Concise (reference style guides, don't embed them)
- Durable (core workflow aid, not a temporary migration helper)

**Anti-patterns:**
- Triggers on common actions (docstrings, formatting) - see [pytorch docstring skill](https://github.com/pytorch/pytorch/blob/43c30f607eeca0d3e9a26911d9c2131fc250eadd/.claude/skills/docstring/SKILL.md)
- Narrow scope (single API conversion) - see [pytorch AT_DISPATCH_V2 skill](https://github.com/pytorch/pytorch/blob/3ca216ae172e35adde34a319a1a01faaf218e7c5/.claude/skills/add-uint-support/SKILL.md)
- Embeds reference material that belongs in `--help` or external docs

**Prefer instead:**
- `~/.claude/skills/` for personal skills
- [Plugins](https://code.claude.com/docs/en/plugins) for sharing with small groups

## Commands

**Bar: Daily workflow automation. Graduated from manual processes.**

Commands should encode workflows that engineers do manually today. They must be:
- Used frequently (not one-off investigations)
- Stable (the underlying tools/processes are mature)
- Broadly applicable (useful to most engineers, not specialists)

**Prefer instead:**
- `~/.claude/commands/` for personal commands
- [Plugins](https://code.claude.com/docs/en/plugins) for sharing with small groups

## Hooks

Hooks auto-approve or modify Claude's behavior. They must be:
- Security-conscious (document threat model)
- Minimal (approve specific tools, not broad categories)
- Self-documenting (embed rationale in the script)

## Summary

| Location | Bar | Examples |
|----------|-----|----------|
| `CLAUDE.md` (checked in) | All devs, every use | None yet |
| `.claude/skills/` | Nearly all devs, durable | `iree-lit-tools` |
| `.claude/commands/` | Daily workflows | `/iree-ci-triage`, `/iree-lit-test` |
| `~/.claude/` | Personal | Your preferences |
| Plugins | Small groups | Team-specific workflows |

When in doubt, keep it local. Graduate to checked-in only after proving value across the team.

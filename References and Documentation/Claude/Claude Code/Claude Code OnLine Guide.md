# Claude Code OnLine Guide

Source: https://www.siddharthbharath.com/claude-code-the-complete-guide/

---

## Hierarchical CLAUDE.md Files

Claude Code supports multiple CLAUDE.md files in a hierarchy, allowing you to organize knowledge at different levels of specificity. This is really helpful to manage context if your files and codebase become too large.

A pattern I use is my primary Claude.md file for my project, and then a specific file for the frontend and backend, like so:

**File Structure Example:**
- Primary CLAUDE.md (project-level)
- Frontend CLAUDE.md (frontend-specific)
- Backend CLAUDE.md (backend-specific)

You can also set up a global Claude file that applies to all projects on your computer. This is where you can set personal preferences about the way you code or work.

### How Claude Processes the Hierarchy:

- Claude reads all applicable CLAUDE.md files when starting
- More specific files override general ones
- All relevant context is combined automatically
- Claude prioritizes the most specific guidance for each situation

---

## Additional Documentation

In addition to Claude files, I also set up project documentation files and put them into a 'docs' folder. This is where I put my initial PRD, and other files for architecture, design principles, database schemas, and so on. Then, in my Claude.md file, I point to the documentation:

This allows us to separate what goes into the prompt and fills up our context window (the Claude.md files) and what stays outside until it needs to be referenced.

---

## Managing Context

If you think the project or feature might be too big for one context window, ask Claude to break it down into a project plan and save it to a markdown file. Then, ask Claude to pick off the first part and finish it in one chat. When that's done, tell Claude to update the plan, clear the chat, and ask it to reference the plan and continue from there.

If you do get to a point where you're running out of context but can't clear it all yet, you can use `/compact` with instructions on what to save.

---

## Deploying Sub-agents

Each sub-agent maintains its own conversation history and context, so your main chat with Claude doesn't get filled with the context of the sub-agent. You can also limit their access to certain tools.

Just use the `/agents` command, follow the instructions, and tell Claude what kind of sub-agent you need.

---

## Git Branches

To do that, we'll use Git to ensure Claude doesn't mess with our core code. I won't explain what it is or how it works (not in scope for this tutorial) but suffice it to say it's not as scary as it sounds and Claude will help you.

Here's what you do:

1. Every time you want to start a new project or feature, ask Claude to create a new branch first. This basically puts you in a new "version" of your code so that any changes you make are isolated to this branch and don't impact the main branch. This means if you mess everything up, you can simply switch back to main and delete this branch.

2. When Claude is done, ask it to test the app. We will get into testing strategies later but for now let Claude do it's default testing. You should also run the app yourself to see if there are any errors.

3. If it all looks good, have Claude update the documentation if needed (as I mentioned earlier), and then ask it to commit changes.

4. If this is a multi-part feature, repeat the above steps. When you're done and satisfied with everything, tell Claude to merge it back into the main branch.

---

## Git Worktrees

Git worktrees let you check out multiple branches simultaneously, each in its own directory. Combined with Claude Code, this means:

- Multiple Claude instances can work on different features in parallel
- Each Claude maintains its own conversation context and project understanding
- No context switching overhead or lost momentum
- True parallel development without conflicts

**Here's what the structure might look like:**

```
project-root/
├── main/          (main branch worktree)
├── feature-a/     (feature-a branch worktree)
└── feature-b/     (feature-b branch worktree)
```

---

## Custom Slash Commands

### Hooks – Deterministic Automation

Hooks are user-defined shell commands that execute automatically at specific points in Claude Code's lifecycle. They provide guaranteed automation that doesn't rely on Claude "remembering" to do something.

You can set up hooks by typing in `/hooks`. You'll be asked to select one of the options from above. If you select pre or post tool use, you'll have to specify which tool first before you add your hook.


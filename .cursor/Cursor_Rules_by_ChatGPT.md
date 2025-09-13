"Cursor – Rules"
by ChatGPT

Here’s what I found on the **Cursor Rules** page:

---

### Overview: What Are Cursor Rules?

Cursor **Rules** allow you to define reusable, persistent instructions for the AI Agent. Think of them like guided system-level prompts—consistent context that shapes how the Agent generates code, applies edits, or handles workflows. Once defined, rules are integrated at the *start* of each relevant model context.

There are four main types of rules in Cursor:

1. **Project Rules**

   * Reside in `.cursor/rules` files within your project.
   * Version-controlled and scoped to specific parts of your codebase.
   * Useful for encapsulating domain-specific knowledge, templates, architectural guidelines, or enforcing style conventions.
     

2. **User Rules**

   * Global rules defined in your Cursor settings.
   * Apply across all your projects.
   * Perfect for setting personal preferences, e.g. writing style or coding conventions.
     

3. **AGENTS.md**

   * A simple markdown file placed in your project root.
   * Serves as an alternative to `.cursor/rules`, particularly for lightweight use cases.
   * Lacks metadata and advanced scoping features but is easy to read and manage.
     

4. **.cursorrules (Legacy)**

   * Old format supported for backward compatibility.
   * Recommended to migrate away in favor of Project Rules.
     

---

### Rule Anatomy & Behavior

* **How Rules Work**: Since AI models don’t retain memory between sessions, rules provide persistent context at the *prompt* level. They help maintain consistent behavior across interactions.
  

* **Types & Scoping**:

  * **Always**: Always included.
  * **Auto Attached**: Included when referencing files matching specific patterns.
  * **Agent Requested**: Available for the AI to include if relevant (must include a description).
  * **Manual**: Only included when explicitly invoked with `@ruleName`.
    

* **Nested Rules**: You can organize rules within nested `.cursor/rules` directories to scope them to particular subfolders. They will auto-attach when files in those directories are referenced.
  

* **Creating Rules**: Use the “New Cursor Rule” command or go to *Cursor Settings → Rules* to create and manage rules.
  

---

### Best Practices & Examples

* **Keep rules focused** (under 500 lines).
* **Use composable, scoped rules** instead of large monoliths.
* Include **concrete examples** or file references.
* Avoid vague or overly generic guidance.
* Reuse common patterns across your projects.
  

Examples highlighted include:

* Enforcing frontend component patterns (e.g., using Tailwind, component naming, Framer Motion for animations).
* Using `zod` for API validation schemas.
* Templates for Express APIs or React component structure.
* Automating documentation generation or specific development workflows.
  

---

### Summary Table

| Rule Type                 | Location/Scope                 | Use Case                                 |
| ------------------------- | ------------------------------ | ---------------------------------------- |
| Project Rules             | `.cursor/rules` subdirectories | Scoped templates, coding conventions     |
| User Rules                | Global settings                | Personal preferences or style rules      |
| AGENTS.md                 | Project root                   | Lightweight, markdown-based instructions |
| .cursorrules     (legacy) | Deprecated legacy support      |                                          |

---

### TL;DR

Cursor “Rules” let you define stable guidance for your AI across projects. You can define:

* **Project Rules** for scoped team-wide behaviors,
* **User Rules** for personal preferences,
* **AGENTS.md** for simpler instruction sets,
  and avoid using the deprecated `.cursorrules`.



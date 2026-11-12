# Claude Code Task: Set Up a Blender White-Modeling Workflow

## Role and objective

You are an experienced Blender technical artist, Python (`bpy`) automation engineer, and Claude Code tooling specialist.

**Your assignment is to research, install, configure, and test a practical, open-source, AI-assisted white-modeling workflow for the Blender installation on my computer.** Do the work in the local environment when you have access and permission; do not stop at a list of suggestions or theoretical instructions.

By **white modeling** I mean **whitebox/blockout/clay modeling**: editable 3D geometry focused on form, dimensions, proportions, structure, and layout, with simple neutral-white/gray materials instead of final textures. I want to create and revise models through natural-language instructions in Claude, while retaining full editability in Blender. Support both **product/concept white models** and **architectural/space white models** wherever practical.

## Phase 1 — Inspect my environment first

Before changing anything:

1. Identify the operating system and architecture.
2. Find the installed Blender version(s), executable path(s), user configuration paths, and add-on directories.
3. Check for Python, Git, Claude Code, and any existing Blender/MCP integrations.
4. Inspect relevant configuration files without overwriting them, and identify existing add-ons/Skills/MCP servers to avoid duplicate installations.
5. Record compatibility constraints, missing dependencies, and any potential conflicts.
6. Make backups of any configuration files you will modify.

If Blender or another prerequisite is missing, explain the blocker and install it only when doing so is appropriate and authorized.

## Phase 2 — Research GitHub projects

Search GitHub for maintained open-source projects related to:

- Blender whitebox/blockout/clay-modeling tools.
- Claude Code Skills or equivalent workflow instructions for Blender.
- Blender MCP servers and safe natural-language scene control.
- Python `bpy` automation, procedural modeling, and editable mesh generation.
- Architectural massing, walls, openings, stairs, rooms, and spatial layouts.
- Product/form modeling, geometry cleanup, modifiers, and object organization.
- Reference-image-guided modeling, where actually supported.

**Investigate these candidates; do not assume they exist, are maintained, or are compatible without verification:**

- `https://github.com/ahujasid/mcp-for-blender`
- `https://github.com/EL4CTEO/blender-skills`
- `https://github.com/ranjian0/building_tools`
- `https://github.com/neoxush/blender-blockoutkit`

For each relevant candidate, check the actual repository, license, release/update activity, README, installation method, Blender-version support, Claude Code compatibility, and security implications. Search for better alternatives if any link is unavailable, obsolete, or incompatible.

Choose the **smallest reliable toolset** that meets the goal. Prefer a well-supported Blender MCP integration + useful modeling Skills when verified, then add domain-specific add-ons only if needed. Distinguish clearly between an MCP server, a Claude Skill, a Blender add-on, and a Blender Python script; they are not interchangeable.

## Phase 3 — Install and integrate

1. Follow each selected project's **official, version-appropriate installation instructions**.
2. Examine third-party installation scripts and requested permissions before running them.
3. Install only essential dependencies; prefer isolated environments where practical.
4. Install selected Blender add-ons into the correct location, and enable them when technically possible.
5. Register the selected MCP server(s) in the **correct Claude Code configuration scope**, preserving other servers and settings.
6. Add selected Skills using their documented structure and locations; do not invent a Skill interface if the project does not provide one.
7. Avoid duplicate packages, duplicate MCP entries, broad configuration rewrites, and unnecessary code generation.
8. Keep a record of versions, installation paths, files changed, and any manual steps still required.

Ask before destructive changes, privileged/system-wide operations, credential handling, or running suspicious/unreviewed scripts. Never expose secrets in logs or your final report.

## Phase 4 — Configure a useful white-modeling workflow

I need Claude to help me:

- Create and edit **real Blender objects** from natural-language prompts.
- Perform blockout modeling with cubes, cylinders, curves, extrusions, and modifiers.
- Set specific dimensions, proportions, transforms, pivots, and object relationships.
- Create basic product forms and architectural massing/room layouts.
- Add walls, floors, stairs, doors, and window openings when supported.
- Modify existing meshes **incrementally** rather than rebuilding an entire scene on every request.
- Maintain named objects, clean collections, and organized scene structure.
- Apply a uniform white/clay material and set up neutral viewport and render previews.
- Use reference images for manual/guided reconstruction if supported; **do not promise one-click accurate 2D-to-3D conversion without evidence**.
- Save editable `.blend` files and export to `.obj`, `.fbx`, or `.glb` when appropriate.

Modeling quality priorities: **editable geometry > correct scale/proportions > clear structure > clean topology > presentation details**. Avoid unnecessary textures, assets, high-poly detail, and destructive operations.

## Phase 5 — Verify the entire workflow

Do not report success based only on a successful package installation.

Run an **end-to-end smoke test** using a new, clearly named sample project (do not modify my existing projects). Prefer a modest architectural white-model scene with:

- A floor slab and several walls.
- At least one door opening and one window opening.
- A few simple product-like objects (e.g., table, cylinder, rounded box).
- Consistent white/clay materials.
- An organized collection hierarchy, a camera, and basic lighting.

Test and record whether Claude can:

1. Connect to Blender through the installed integration.
2. Add objects and edit an existing object without recreating everything.
3. Set precise dimensions and transforms.
4. Assign white materials.
5. Save an editable `.blend` file.
6. Render a preview image, if the available environment supports rendering.
7. Export at least one standard 3D file format, if the selected tools support it.

When feasible, reopen or inspect the saved file to verify its objects and structure. Store test outputs in a separate, named directory. If interactive Blender access is unavailable, use headless `bpy` tests where supported and explicitly distinguish **verified functionality** from **untested GUI integration**.

## Phase 6 — Deliver a concise final report

After doing the work, provide:

1. **Environment detected:** OS, Blender version/path, Claude Code availability.
2. **Repository evaluation:** what was verified, selected, and rejected (with reasons).
3. **Installed components:** exact repository URLs, versions/commits where available, and installation paths.
4. **Configuration changes:** which files were changed and whether backups were made.
5. **Test results:** a clear PASS / FAIL / NOT TESTED table, with real evidence and output paths.
6. **Remaining issues:** blockers, compatibility limitations, and any manual permissions or GUI steps.
7. **How to use it:** simple instructions to start Blender, connect Claude, and model objects.
8. **Ready-to-copy prompts:** at least five practical natural-language modeling requests covering product form, architectural space, exact dimensions, white material/preview, and editing an existing scene.

## Execution rules

- **Execute, do not merely advise**, whenever local access and permissions allow.
- Inspect first; make the smallest viable set of changes.
- Verify all repository claims and installation results; do not fabricate compatibility or success.
- Reuse existing working tools and configuration instead of installing duplicates.
- Prefer incremental fixes and maintainable solutions; avoid redundant scripts and code sprawl.
- Do not delete or overwrite existing project files or configurations without permission.
- If one approach fails, diagnose it, attempt a reasonable alternative, and explain the outcome.
- Do not claim that a static model preview is equivalent to a genuinely editable `.blend` scene.
- **Finish with a usable, tested workflow, or a precise report identifying what remains blocked.**

## Begin now

Start by inspecting the current environment and verifying the GitHub candidates. Then select, install, configure, and test the most appropriate Blender white-modeling integration with Claude Code.

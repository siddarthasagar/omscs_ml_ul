# OMSCS ML UL

This repository is set up as a small unsupervised-learning workspace for the OMSCS ML course. The default workflow is now spec-driven: define the work, acceptance criteria, and validation plan before changing code or writing an analysis.

## Spec-Driven Workflow

1. Create a spec folder.
   `make spec-new NAME=baseline-kmeans-study`
2. Fill in `spec.md`.
   Capture the problem, scope, non-goals, acceptance criteria, validation plan, and open questions.
3. Break the work into `tasks.md`.
   Keep tasks ordered, testable, and small enough to complete without reinterpretation.
4. Implement only after the spec is stable.
   Code changes should map back to an acceptance criterion or a listed task.
5. Validate and close the loop.
   Run the planned checks, update the task list, and record any scope changes in the spec.

## Layout

`specs/templates/`
Spec and task templates used by the scaffold command.

`specs/<YYYYMMDD>-<slug>/spec.md`
The source of truth for the requested change, experiment, or analysis.

`specs/<YYYYMMDD>-<slug>/tasks.md`
The execution checklist derived from the spec.

`scripts/new_spec.py`
Creates a new spec folder from the templates.

`scripts/check_specs.py`
Validates that each spec folder has the required files and section headings.

## Commands

`make spec-new NAME=<slug>`
Create a new spec from the repo templates.

`make spec-list`
List tracked spec folders.

`make spec-check`
Validate spec structure and required headings.

`make test`
Run automated tests for the repository tooling.

## Working Agreement

- Start with the spec when the task changes behavior, adds analysis scope, or introduces new project structure.
- Keep non-goals explicit so experiments do not expand silently.
- Treat acceptance criteria as the contract for completion.
- If implementation diverges from the spec, update the spec first, then continue.

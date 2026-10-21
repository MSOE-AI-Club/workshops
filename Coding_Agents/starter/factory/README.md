# The factory floor

- `current/` holds the run in progress: `TICKET.md` (intake), `SPEC.md` (shape and build), `REVIEW.md` (audit). Only one run at a time.
- `done/` holds finished runs, one numbered folder each (`001-build-snake`, ...). `factory-close` moves `current/` here.
- `BACKLOG.md` has ideas you can feed to `$factory-intake`.
- `skill-template.md` is the starting point for a new skill, like the audit station you'll write.

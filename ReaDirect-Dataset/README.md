# Content Bank Export

These files are Git-tracked copies of editable content from `database/seed-data/readirect/`.

## Subfolders

- `assessment/` contains diagnostic assessment item banks, reading passages, and comprehension questions.
- `modules/` contains module activity item banks and activity selection rules.
- `agents/` contains fixed-agent scripts and agent commentary templates.
- `rules/` contains classification and placement rules that are represented as seed CSVs.
- `feedback/` contains feedback template CSVs.
- `prompts/` is reserved for LLM prompt template exports when prompt CSVs are added.
- `docs/` contains copied seed-data documentation.

## Editing Rules

- Keep CSV headers unchanged unless the importer/seeder is also updated.
- Use stable keys such as `module_1`, `module_2`, `module_3`, and content type keys already used by seeders.
- Coordinate large CSV edits on branches to avoid merge conflicts.
- Do not add private learner data, audio, API keys, SQL dumps, or `.env` values.

To refresh this export from the current seed source, run:

```powershell
.\scripts\export-content-bank.ps1
```

# Content Bank

This directory is for safe ReaDirect content-bank files copied from the main ReaDirect repository or from `ReaDirect-Dataset/`.

It may contain safe CSVs for prompts, expected answers, accepted answers, module tags, activity types, rules, and feedback templates. These files can be tracked when they contain instructional content only.

Do not place learner submissions, learner audio, private metadata, API keys, SQL dumps, or `.env` files here.

Supported import patterns:

- Copy safe CSV folders directly into `content_bank/`.
- Import a Laravel-generated ZIP with `python scripts/import_content_bank_zip.py --zip-path path/to/export.zip`.
- Keep large archives, audio, SQL dumps, and secrets out of this directory.

The content loader supports both:

- `content_bank/assessment`, `content_bank/modules`, `content_bank/rules`, etc.
- `content_bank/readirect-content-bank/assessment`, `modules`, `rules`, etc.

Phase 6 enrichment writes generated review outputs to `content_bank_enriched/`. Do not overwrite source content-bank CSVs directly.

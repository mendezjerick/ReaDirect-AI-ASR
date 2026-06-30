# ReaDirect Phase 3 Module Seed Data

These CSV files are development item banks for the Phase 3 learning module flow. They are not fixed learner screens. A module attempt should select active items from these banks, save the selected rows into `module_attempt_items`, and reuse that locked sequence for the same attempt.

## Files

- `module1_letter_sound_activities.csv` contains Module 1 letter and sound practice plus mastery-check items.
- `module2_word_reading_activities.csv` contains Module 2 word reading practice plus mastery-check items.
- `module3_sentence_fluency_activities.csv` contains Module 3 sentence reading and fluency practice plus mastery-check items.
- `module_feedback_templates.csv` contains reusable, child-friendly Coach + Feedback Agent templates. These are template-based only and do not call an LLM.
- `module_activity_selection_rules.csv` defines how many active items to lock per activity type and how many mastery-check items to lock.

## Selection Rules

Module CSVs are item banks. When a learner starts or resumes a module, the system creates or reuses a `module_attempt`. Practice activities and mastery checks should lock selected rows into `module_attempt_items` with a `prompt_snapshot`. This protects old attempts if seed content changes later.

Practice activities may select a small set of active items for the requested activity type. Mini mastery checks select 10 active mastery items for the module attempt. The UI should read from locked `module_attempt_items`, not directly from random item-bank queries.

## Module Bank Sync

The module activity banks mirror the Laravel seed-data source of truth and use lesson-specific activity keys.

- Module 1 uses four lesson segments plus `mastery_check`: `letter_pair_identification`, `highlighted_first_letter`, `first_letter_identification`, and `missing_first_letter`.
- Module 2 uses four lesson segments plus `mastery_check`: `display_word_reading`, `split_word_reading`, `highlighted_rhyme_word`, and `highlighted_sentence_word`.
- Module 3 uses four lesson segments plus `mastery_check`: `simple_sentence_reading`, `comma_pause_reading`, `full_stop_pause_reading`, and `mixed_punctuation_fluency`.
- `module_activity_selection_rules.csv` keeps four active lesson boxes per module and separate module mastery checks.

Run `python scripts/generate_module_datasets.py` from the repository root to sync the module banks from the Laravel seed-data folder. Run `python scripts/validate_module_datasets.py` to check row counts, required columns, prompt IDs, activity segment counts, mastery flags, and accepted-answer alignment.

## Feedback and Decisions

Coach feedback is template-based for Phase 3. Official mastery decisions must use `ModuleMasteryService` only:

- Module 1: `>= 90` move to Module 2, `60-89` repeat Module 1, `< 60` extra phoneme drills.
- Module 2: `>= 90` move to Module 3, `60-89` repeat Module 2, `< 60` return to Module 1.
- Module 3: `>= 90` proceed to final reassessment placeholder, `70-89` repeat Module 3, `< 70` return to Module 2.

LLM feedback can be added later through a separate service boundary. These seed files are original sample content and can later be replaced with official ARAL-aligned content.

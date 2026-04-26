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

## Module Bank Expansion

The module activity banks were expanded to 500 active rows per module CSV while preserving the existing headers, module keys, activity type names, mastery flags, and selection-rule behavior.

- Module 1 uses five 100-row segments: `hear_and_repeat`, `see_letter_say_sound`, `match_sound_to_letter`, `sound_drill`, and `mastery_check`.
- Module 2 uses five 100-row segments: `read_word`, `word_family_drill`, `minimal_pair`, `word_accuracy_challenge`, and `mastery_check`.
- Module 3 uses five 80-row practice segments plus a 100-row mastery segment: `read_sentence`, `read_with_coach`, `timed_sentence_reading`, `pause_practice`, `fluency_challenge`, and `mastery_check`.
- `module_activity_selection_rules.csv`, the `rules` folder, mastery decisions, scoring rules, agents, prompts, feedback logic, and runtime behavior were not changed.

Run `python scripts/generate_module_datasets.py` from the repository root to regenerate the expanded module banks. Run `python scripts/validate_module_datasets.py` to check row counts, required columns, IDs, sequences, activity segment counts, mastery flags, active flags, and accepted-answer alignment.

## Feedback and Decisions

Coach feedback is template-based for Phase 3. Official mastery decisions must use `ModuleMasteryService` only:

- Module 1: `>= 90` move to Module 2, `60-89` repeat Module 1, `< 60` extra phoneme drills.
- Module 2: `>= 90` move to Module 3, `60-89` repeat Module 2, `< 60` return to Module 1.
- Module 3: `>= 90` proceed to final reassessment placeholder, `70-89` repeat Module 3, `< 70` return to Module 2.

LLM feedback can be added later through a separate service boundary. These seed files are original sample content and can later be replaced with official ARAL-aligned content.

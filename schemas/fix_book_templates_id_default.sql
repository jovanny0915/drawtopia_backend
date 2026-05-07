-- Ensure book_templates.id is auto-populated.
--
-- Some environments created the `book_templates` table without a default value
-- on the `id` column, which leads to errors like:
--   "null value in column \"id\" of relation \"book_templates\"
--    violates not-null constraint"
-- when an INSERT is performed without explicitly providing an id.
--
-- This migration:
--   1. Enables pgcrypto so gen_random_uuid() is available.
--   2. Backfills any existing rows that have a null id.
--   3. Sets a default of gen_random_uuid() on the column going forward.
--   4. Reasserts the NOT NULL constraint.

CREATE EXTENSION IF NOT EXISTS pgcrypto;

UPDATE book_templates
SET id = gen_random_uuid()
WHERE id IS NULL;

ALTER TABLE book_templates
    ALTER COLUMN id SET DEFAULT gen_random_uuid();

ALTER TABLE book_templates
    ALTER COLUMN id SET NOT NULL;

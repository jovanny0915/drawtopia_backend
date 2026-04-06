-- Migration: add_template_id_to_stories.sql
-- Adds a nullable text column `template_id` to the `stories` table.

ALTER TABLE public.stories
ADD COLUMN IF NOT EXISTS template_id text;

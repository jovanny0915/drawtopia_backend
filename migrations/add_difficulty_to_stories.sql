-- Migration: add_difficulty_to_stories.sql
-- Adds a nullable text column `difficulty` to the `stories` table.

ALTER TABLE public.stories
ADD COLUMN IF NOT EXISTS difficulty text;

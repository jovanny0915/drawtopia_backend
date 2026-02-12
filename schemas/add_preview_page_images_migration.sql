-- Migration to add preview page images to stories table
-- Run this SQL in your Supabase SQL Editor
-- Adds: copyright_image, last_word_page_image, and back_cover_image

-- Add copyright_image column to stories table if it doesn't exist
ALTER TABLE stories 
ADD COLUMN IF NOT EXISTS copyright_image TEXT;

-- Add last_word_page_image column to stories table if it doesn't exist
ALTER TABLE stories 
ADD COLUMN IF NOT EXISTS last_word_page_image TEXT;

-- Add back_cover_image column to stories table if it doesn't exist
ALTER TABLE stories 
ADD COLUMN IF NOT EXISTS back_cover_image TEXT;

-- Add comments to explain the fields
COMMENT ON COLUMN stories.copyright_image IS 'URL to the copyright page image (displayed on left half of copyright/dedication spread)';
COMMENT ON COLUMN stories.last_word_page_image IS 'URL to the last word page image (displayed after story pages, before back cover)';
COMMENT ON COLUMN stories.back_cover_image IS 'URL to the back cover image (displayed as final page)';

-- Create indexes for better query performance when filtering by these fields
CREATE INDEX IF NOT EXISTS idx_stories_copyright_image ON stories(copyright_image) WHERE copyright_image IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_stories_last_word_page_image ON stories(last_word_page_image) WHERE last_word_page_image IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_stories_back_cover_image ON stories(back_cover_image) WHERE back_cover_image IS NOT NULL;

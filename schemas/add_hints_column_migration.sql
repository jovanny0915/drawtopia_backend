-- Migration to add hints column to stories table
-- This column tracks remaining hints for each story (initial value is 3)
-- Run this SQL in your Supabase SQL Editor

-- Add hints column to stories table if it doesn't exist
ALTER TABLE stories 
ADD COLUMN IF NOT EXISTS hints INTEGER DEFAULT 3;

-- Add comment to explain the field
COMMENT ON COLUMN stories.hints IS 'Number of hints remaining for this story. Initial value is 3. Decrements when hints are used during search game.';

-- Create index for better query performance
CREATE INDEX IF NOT EXISTS idx_stories_hints ON stories(hints);

-- Update existing stories to have hints = 3 if null (for existing search type stories)
UPDATE stories 
SET hints = 3 
WHERE hints IS NULL AND story_type = 'search';

-- Update existing stories to have hints = NULL if not a search type story (hints only apply to search games)
UPDATE stories 
SET hints = NULL 
WHERE story_type != 'search' OR story_type IS NULL;

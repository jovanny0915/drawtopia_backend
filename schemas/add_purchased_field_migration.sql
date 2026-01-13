-- Add purchased field to stories table
-- Run this SQL in your Supabase SQL Editor

-- Add purchased column to stories table if it doesn't exist
ALTER TABLE stories 
ADD COLUMN IF NOT EXISTS purchased BOOLEAN DEFAULT FALSE;

-- Add comment to explain the field
COMMENT ON COLUMN stories.purchased IS 'Indicates if this story has been purchased via one-time payment';

-- Create index for better query performance
CREATE INDEX IF NOT EXISTS idx_stories_purchased ON stories(purchased) WHERE purchased = TRUE;

-- Update existing stories to have purchased = false if null
UPDATE stories 
SET purchased = FALSE 
WHERE purchased IS NULL;


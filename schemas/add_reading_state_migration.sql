-- Migration to add reading_state column to stories table
-- This column tracks reading statistics for each story

-- Add reading_state column to stories table
ALTER TABLE stories
ADD COLUMN IF NOT EXISTS reading_state JSONB DEFAULT NULL;

-- Add index for better query performance
CREATE INDEX IF NOT EXISTS idx_stories_reading_state ON stories USING GIN (reading_state);

-- Add comment to explain the column structure
COMMENT ON COLUMN stories.reading_state IS 
'JSON object storing reading statistics. Structure depends on story type:
For story adventure (story_type = "story"): 
{
  "reading_time": <seconds as integer>,
  "audio_listened": <boolean>
}

For interactive search (story_type = "search"):
{
  "reading_time": <seconds as integer>,
  "avg_star": <float>,
  "avg_hint": <float>
}';

-- Example queries:
-- Update reading state for a story:
-- UPDATE stories 
-- SET reading_state = '{"reading_time": 300, "audio_listened": true}'::jsonb
-- WHERE uid = 'story-id';

-- Query stories by reading time:
-- SELECT * FROM stories 
-- WHERE (reading_state->>'reading_time')::int > 300;


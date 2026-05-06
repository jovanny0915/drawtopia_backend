-- Add learning theme metadata to saved stories

ALTER TABLE stories
  ADD COLUMN IF NOT EXISTS learning_theme TEXT;

COMMENT ON COLUMN stories.learning_theme IS 'Display name of the selected learning theme used for story generation';

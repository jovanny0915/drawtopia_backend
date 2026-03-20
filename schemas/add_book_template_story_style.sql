-- Add story_style metadata to book_templates
ALTER TABLE book_templates
ADD COLUMN IF NOT EXISTS story_style TEXT;

COMMENT ON COLUMN book_templates.story_style IS 'Story style for the template (for example: adventure, search-and-find)';

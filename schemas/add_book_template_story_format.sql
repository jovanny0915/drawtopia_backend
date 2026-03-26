-- Add story_format metadata to book_templates (e.g. adventure_story, interactive_story; free-form text)
ALTER TABLE book_templates
ADD COLUMN IF NOT EXISTS story_format TEXT;

COMMENT ON COLUMN book_templates.story_format IS 'Story format for the template (e.g. adventure_story, interactive_story); free-form text';

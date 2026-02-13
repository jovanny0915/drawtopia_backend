-- Add last_admin_page_image to stories (right half of last-words spread)
ALTER TABLE stories
ADD COLUMN IF NOT EXISTS last_admin_page_image TEXT;

COMMENT ON COLUMN stories.last_admin_page_image IS 'URL to the last admin/scene page image (right half of last-words spread, before back cover)';

CREATE INDEX IF NOT EXISTS idx_stories_last_admin_page_image ON stories(last_admin_page_image) WHERE last_admin_page_image IS NOT NULL;

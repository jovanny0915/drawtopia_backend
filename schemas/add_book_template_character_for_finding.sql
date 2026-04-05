-- Add character_for_finding to book_templates (for interactive story templates)
-- Stores a list of URLs (story pages format) for the character used in search-and-find templates.

ALTER TABLE book_templates
ADD COLUMN IF NOT EXISTS character_for_finding TEXT[];

COMMENT ON COLUMN book_templates.character_for_finding IS 'Array of URLs to character-for-finding images (story pages format) for interactive story templates';

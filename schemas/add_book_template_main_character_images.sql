-- Add main_character_images to book_templates (for interactive story templates)
-- Stores a list of URLs (e.g., different poses/expressions) for the main character.

ALTER TABLE book_templates
ADD COLUMN IF NOT EXISTS main_character_images TEXT[];

COMMENT ON COLUMN book_templates.main_character_images IS 'Array of URLs to main character reference images for interactive story templates';


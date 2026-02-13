-- Add last_words_page_image to book_templates (left half of last-words spread)
ALTER TABLE book_templates
ADD COLUMN IF NOT EXISTS last_words_page_image TEXT;

COMMENT ON COLUMN book_templates.last_words_page_image IS 'URL to last words page image (left half of final spread before back cover)';

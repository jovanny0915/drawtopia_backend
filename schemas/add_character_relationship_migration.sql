-- Migration: Add character relationship to stories table
-- This allows stories to reference characters instead of duplicating data
-- Run this AFTER creating the characters table

-- Add character_id foreign key to stories table
ALTER TABLE stories
ADD COLUMN IF NOT EXISTS character_id BIGINT REFERENCES characters(id) ON DELETE SET NULL;

-- Create index for better query performance
CREATE INDEX IF NOT EXISTS idx_stories_character_id ON stories(character_id);

-- Add trigger to stories table to update character usage stats
CREATE TRIGGER increment_character_usage_on_story_creation
    AFTER INSERT ON stories
    FOR EACH ROW
    EXECUTE FUNCTION increment_character_usage();

-- Comments for documentation
COMMENT ON COLUMN stories.character_id IS 'Reference to the character used in this story. NULL means character data is stored directly in story fields (legacy stories)';

-- Note: Existing stories will have character_id = NULL
-- You can optionally migrate existing stories to create character records:
-- 
-- Example migration script (run separately if needed):
-- 
-- WITH inserted_characters AS (
--     INSERT INTO characters (
--         user_id, 
--         child_profile_id, 
--         character_name, 
--         character_type, 
--         special_ability,
--         character_style,
--         original_image_url,
--         enhanced_images,
--         age_group
--     )
--     SELECT DISTINCT ON (user_id, character_name, original_image_url)
--         user_id,
--         child_profile_id,
--         character_name,
--         character_type,
--         special_ability,
--         character_style,
--         original_image_url,
--         enhanced_images,
--         NULL as age_group
--     FROM stories
--     WHERE character_id IS NULL 
--         AND character_name IS NOT NULL 
--         AND original_image_url IS NOT NULL
--     RETURNING id, user_id, character_name, original_image_url
-- )
-- UPDATE stories s
-- SET character_id = ic.id
-- FROM inserted_characters ic
-- WHERE s.user_id = ic.user_id 
--     AND s.character_name = ic.character_name 
--     AND s.original_image_url = ic.original_image_url
--     AND s.character_id IS NULL;


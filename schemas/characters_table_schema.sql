-- Characters Table Schema for Drawtopia
-- Run this SQL in your Supabase SQL Editor
-- This table stores character information separately from stories,
-- allowing characters to be reused across multiple books

-- Create characters table
CREATE TABLE IF NOT EXISTS characters (
    id BIGSERIAL PRIMARY KEY,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    -- Ownership and relationship
    user_id UUID NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
    child_profile_id BIGINT REFERENCES child_profiles(id) ON DELETE SET NULL,
    
    -- Basic character information
    character_name VARCHAR(255) NOT NULL,
    character_type VARCHAR NOT NULL CHECK (character_type IN ('person', 'animal', 'magical_creature')),
    special_ability TEXT,
    
    -- Character appearance/style
    character_style VARCHAR NOT NULL CHECK (character_style IN ('3d', 'cartoon', 'anime')),
    
    -- Character images
    original_image_url TEXT NOT NULL,
    enhanced_images TEXT[], -- Array of enhanced/processed character image URLs
    thumbnail_url TEXT, -- Small thumbnail for quick display in character selection
    
    -- Character metadata
    age_group VARCHAR CHECK (age_group IN ('3-6', '7-10', '11-12')),
    description TEXT, -- Optional description of the character
    
    -- Usage statistics
    times_used INTEGER DEFAULT 0, -- How many stories have used this character
    last_used_at TIMESTAMP WITH TIME ZONE,
    
    -- Character status
    is_active BOOLEAN DEFAULT true, -- Can be used to soft-delete characters
    is_favorite BOOLEAN DEFAULT false, -- User can mark favorite characters
    
    -- Character extraction data (for AI generation consistency)
    extraction_data JSONB, -- Stores detailed character features extracted by AI
    -- Example structure:
    -- {
    --   "facial_features": { "eye_color": "blue", "hair_color": "brown" },
    --   "clothing": ["red shirt", "blue jeans"],
    --   "distinctive_features": ["glasses", "freckles"],
    --   "pose": "standing",
    --   "extraction_model": "gemini-2.5",
    --   "extraction_timestamp": "2026-01-02T..."
    -- }
    
    -- Tags for categorization and search
    tags TEXT[] -- e.g., ['superhero', 'brave', 'funny']
);

-- Create indexes for better performance
CREATE INDEX IF NOT EXISTS idx_characters_user_id ON characters(user_id);
CREATE INDEX IF NOT EXISTS idx_characters_child_profile_id ON characters(child_profile_id);
CREATE INDEX IF NOT EXISTS idx_characters_character_type ON characters(character_type);
CREATE INDEX IF NOT EXISTS idx_characters_character_style ON characters(character_style);
CREATE INDEX IF NOT EXISTS idx_characters_is_active ON characters(is_active);
CREATE INDEX IF NOT EXISTS idx_characters_is_favorite ON characters(is_favorite);
CREATE INDEX IF NOT EXISTS idx_characters_created_at ON characters(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_characters_last_used_at ON characters(last_used_at DESC);
CREATE INDEX IF NOT EXISTS idx_characters_times_used ON characters(times_used DESC);

-- GIN index for JSONB extraction_data for fast querying
CREATE INDEX IF NOT EXISTS idx_characters_extraction_data ON characters USING GIN (extraction_data);

-- GIN index for tags array for fast tag-based searches
CREATE INDEX IF NOT EXISTS idx_characters_tags ON characters USING GIN (tags);

-- Enable Row Level Security (RLS)
ALTER TABLE characters ENABLE ROW LEVEL SECURITY;

-- RLS Policies

-- Policy: Users can view their own characters
CREATE POLICY "Users can view own characters" ON characters
    FOR SELECT USING (auth.uid() = user_id);

-- Policy: Users can insert their own characters
CREATE POLICY "Users can insert own characters" ON characters
    FOR INSERT WITH CHECK (auth.uid() = user_id);

-- Policy: Users can update their own characters
CREATE POLICY "Users can update own characters" ON characters
    FOR UPDATE USING (auth.uid() = user_id);

-- Policy: Users can delete their own characters
CREATE POLICY "Users can delete own characters" ON characters
    FOR DELETE USING (auth.uid() = user_id);

-- Policy: Service role has full access (for backend operations)
CREATE POLICY "Service role full access to characters" ON characters
    FOR ALL USING (auth.role() = 'service_role');

-- Function to update updated_at timestamp
CREATE OR REPLACE FUNCTION update_characters_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Trigger to auto-update updated_at
CREATE TRIGGER update_characters_updated_at_trigger 
    BEFORE UPDATE ON characters
    FOR EACH ROW 
    EXECUTE FUNCTION update_characters_updated_at();

-- Function to increment times_used and update last_used_at
CREATE OR REPLACE FUNCTION increment_character_usage()
RETURNS TRIGGER AS $$
BEGIN
    -- Update the character's usage stats when a new story is created
    IF NEW.character_id IS NOT NULL THEN
        UPDATE characters
        SET times_used = times_used + 1,
            last_used_at = NOW()
        WHERE id = NEW.character_id;
    END IF;
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Grant permissions
GRANT ALL ON characters TO authenticated;
GRANT USAGE, SELECT ON SEQUENCE characters_id_seq TO authenticated;

-- Comments for documentation
COMMENT ON TABLE characters IS 'Stores character information for reuse across multiple stories';
COMMENT ON COLUMN characters.extraction_data IS 'Detailed character features extracted by AI for consistency across stories';
COMMENT ON COLUMN characters.enhanced_images IS 'Array of enhanced character images in different poses/styles';
COMMENT ON COLUMN characters.times_used IS 'Counter for how many stories have used this character';
COMMENT ON COLUMN characters.tags IS 'User-defined tags for categorization and search';


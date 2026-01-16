-- Search Game Results Table Schema
-- This table stores search game results for interactive search stories
-- Run this SQL in your Supabase SQL Editor

-- Create search_game_results table
CREATE TABLE IF NOT EXISTS search_game_results (
    id BIGSERIAL PRIMARY KEY,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    -- Character relationship
    character_id BIGINT REFERENCES characters(id) ON DELETE CASCADE,
    
    -- Story relationship (optional, for reference)
    story_id BIGINT REFERENCES stories(id) ON DELETE SET NULL,
    
    -- Result data (JSON array of scene results)
    result JSONB NOT NULL DEFAULT '[]'::jsonb,
    -- Structure: [
    --   {
    --     "scene_index": 0,
    --     "scene_title": "The Magical Forest",
    --     "time": "2:30",
    --     "hint_used": 1,
    --     "star_rate": 3
    --   },
    --   ...
    -- ]
    
    -- Summary statistics (individual columns)
    total_time INTEGER NOT NULL DEFAULT 0, -- Total time in seconds
    avg_stars NUMERIC(3, 2) NOT NULL DEFAULT 0.00, -- Average stars (0.00 to 3.00)
    hints_used INTEGER NOT NULL DEFAULT 0, -- Total hints used
    best_scene VARCHAR(255), -- Best scene title (scene with highest stars)
    
    -- Additional metadata
    user_id UUID REFERENCES auth.users(id) ON DELETE CASCADE,
    child_profile_id BIGINT REFERENCES child_profiles(id) ON DELETE SET NULL
);

-- Create indexes for better query performance
CREATE INDEX IF NOT EXISTS idx_search_game_results_character_id ON search_game_results(character_id);
CREATE INDEX IF NOT EXISTS idx_search_game_results_story_id ON search_game_results(story_id);
CREATE INDEX IF NOT EXISTS idx_search_game_results_user_id ON search_game_results(user_id);
CREATE INDEX IF NOT EXISTS idx_search_game_results_created_at ON search_game_results(created_at DESC);

-- Create GIN index for JSONB result column for efficient JSON queries
CREATE INDEX IF NOT EXISTS idx_search_game_results_result ON search_game_results USING GIN (result);

-- Add comments for documentation
COMMENT ON TABLE search_game_results IS 'Stores search game results for interactive search stories';
COMMENT ON COLUMN search_game_results.character_id IS 'Reference to the character used in the search game';
COMMENT ON COLUMN search_game_results.story_id IS 'Reference to the story (optional, for linking results to stories)';
COMMENT ON COLUMN search_game_results.result IS 'JSON array containing detailed results for each scene: [{scene_index, scene_title, time, hint_used, star_rate}, ...]';
COMMENT ON COLUMN search_game_results.total_time IS 'Total time spent in seconds';
COMMENT ON COLUMN search_game_results.avg_stars IS 'Average stars earned across all scenes (0.00 to 3.00)';
COMMENT ON COLUMN search_game_results.hints_used IS 'Total number of hints used across all scenes';
COMMENT ON COLUMN search_game_results.best_scene IS 'Title of the scene with the highest star rating';

-- Example query to get results for a character:
-- SELECT * FROM search_game_results WHERE character_id = 1 ORDER BY created_at DESC;

-- Example query to get average performance across all games:
-- SELECT 
--   character_id,
--   AVG(avg_stars) as avg_stars_overall,
--   AVG(total_time) as avg_time_overall,
--   COUNT(*) as total_games
-- FROM search_game_results
-- GROUP BY character_id;

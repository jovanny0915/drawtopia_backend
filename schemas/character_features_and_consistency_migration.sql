-- Character Features and Consistency Validation Tables
-- Run this SQL in your Supabase SQL Editor
-- Requires: characters and stories tables must exist first.
-- Supports Vision API extraction results and character comparison logging

-- =============================================================================
-- character_features: stores Vision API extraction results per character/image
-- =============================================================================
CREATE TABLE IF NOT EXISTS character_features (
    id BIGSERIAL PRIMARY KEY,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),

    -- Link to character (optional until we have character_id at extraction time)
    character_id BIGINT REFERENCES characters(id) ON DELETE CASCADE,

    -- Source image (Supabase URL or path) used for extraction
    source_image_url TEXT,

    -- Vision API extraction result (labels, colors, etc.)
    -- Structure: { "labels": [{"description": "...", "score": 0.99}], "dominant_colors": [...], "safe_search": {...}, "extraction_model": "google_vision", "response_time_ms": 120 }
    features_json JSONB NOT NULL,

    -- Which Vision model/flow was used (e.g. label_detection + image_properties)
    extraction_model VARCHAR(128) DEFAULT 'google_vision',

    -- Response time in milliseconds (for monitoring)
    response_time_ms INTEGER
);

-- Indexes for generation workflow lookup
CREATE INDEX IF NOT EXISTS idx_character_features_character_id ON character_features(character_id);
CREATE INDEX IF NOT EXISTS idx_character_features_created_at ON character_features(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_character_features_character_created ON character_features(character_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_character_features_features_gin ON character_features USING GIN (features_json);

COMMENT ON TABLE character_features IS 'Vision API extraction results for character drawings';
COMMENT ON COLUMN character_features.features_json IS 'Structured features: labels, dominant_colors, safe_search, etc.';

-- =============================================================================
-- consistency_validation: log every character comparison with confidence scores
-- =============================================================================
CREATE TABLE IF NOT EXISTS consistency_validation (
    id BIGSERIAL PRIMARY KEY,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),

    -- Optional links to story/character for filtering
    story_id BIGINT REFERENCES stories(id) ON DELETE SET NULL,
    character_id BIGINT REFERENCES characters(id) ON DELETE SET NULL,

    -- Which page in the story (1-based)
    page_number INTEGER NOT NULL,

    -- Comparison result
    similarity_score NUMERIC(5,4) NOT NULL CHECK (similarity_score >= 0 AND similarity_score <= 1),
    is_consistent BOOLEAN NOT NULL,
    confidence NUMERIC(5,4) CHECK (confidence IS NULL OR (confidence >= 0 AND confidence <= 1)),

    -- Optional URLs for debugging (scene vs reference)
    scene_image_url TEXT,
    reference_image_url TEXT,

    -- Extra details (e.g. character_match_details, issues) for analytics
    details_json JSONB
);

-- Indexes for quick lookup during generation workflow
CREATE INDEX IF NOT EXISTS idx_consistency_validation_story_id ON consistency_validation(story_id);
CREATE INDEX IF NOT EXISTS idx_consistency_validation_character_id ON consistency_validation(character_id);
CREATE INDEX IF NOT EXISTS idx_consistency_validation_created_at ON consistency_validation(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_consistency_validation_story_page ON consistency_validation(story_id, page_number);
CREATE INDEX IF NOT EXISTS idx_consistency_validation_flagged ON consistency_validation(is_consistent) WHERE is_consistent = false;

COMMENT ON TABLE consistency_validation IS 'Log of every character consistency comparison with confidence scores';
COMMENT ON COLUMN consistency_validation.similarity_score IS '0.0-1.0 similarity between scene and reference character';
COMMENT ON COLUMN consistency_validation.confidence IS 'Model confidence in the comparison';

-- RLS (optional: enable if you want user-scoped access; service_role bypasses)
ALTER TABLE character_features ENABLE ROW LEVEL SECURITY;
ALTER TABLE consistency_validation ENABLE ROW LEVEL SECURITY;

-- Policy: service role has full access
CREATE POLICY "Service role full access to character_features" ON character_features
    FOR ALL USING (auth.role() = 'service_role');
CREATE POLICY "Service role full access to consistency_validation" ON consistency_validation
    FOR ALL USING (auth.role() = 'service_role');

-- Users can read their own character_features via character ownership (if needed later)
-- Users can read consistency_validation for their stories (if needed later)
GRANT ALL ON character_features TO service_role;
GRANT ALL ON consistency_validation TO service_role;
GRANT USAGE, SELECT ON SEQUENCE character_features_id_seq TO service_role;
GRANT USAGE, SELECT ON SEQUENCE consistency_validation_id_seq TO service_role;

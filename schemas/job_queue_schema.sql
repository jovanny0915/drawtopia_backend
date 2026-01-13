-- Job Queue Schema for Batch Processing System
-- Run this SQL in your Supabase SQL Editor

-- Create book_generation_jobs table
CREATE TABLE IF NOT EXISTS book_generation_jobs (
    id BIGSERIAL PRIMARY KEY,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    book_id BIGINT, -- Reference to stories table (can be NULL if book not created yet)
    job_type VARCHAR NOT NULL CHECK (job_type IN ('interactive_search', 'story_adventure')),
    status VARCHAR DEFAULT 'pending' CHECK (status IN ('pending', 'processing', 'completed', 'failed', 'cancelled')),
    priority INTEGER DEFAULT 5 CHECK (priority >= 1 AND priority <= 10), -- 1 = highest, 10 = lowest
    retry_count INTEGER DEFAULT 0,
    max_retries INTEGER DEFAULT 3,
    error_message TEXT,
    job_data JSONB NOT NULL, -- Contains all job parameters (character info, story params, etc.)
    result_data JSONB, -- Contains final results when completed
    started_at TIMESTAMP WITH TIME ZONE,
    completed_at TIMESTAMP WITH TIME ZONE,
    user_id UUID REFERENCES auth.users(id) ON DELETE CASCADE,
    child_profile_id BIGINT REFERENCES child_profiles(id) ON DELETE CASCADE
);

-- Create job_stages table for tracking individual stage progress
CREATE TABLE IF NOT EXISTS job_stages (
    id BIGSERIAL PRIMARY KEY,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    job_id BIGINT NOT NULL REFERENCES book_generation_jobs(id) ON DELETE CASCADE,
    stage_name VARCHAR NOT NULL CHECK (stage_name IN (
        'character_extraction',
        'enhancement',
        'story_generation',
        'scene_creation',
        'consistency_validation',
        'audio_generation',
        'pdf_creation'
    )),
    status VARCHAR DEFAULT 'pending' CHECK (status IN ('pending', 'processing', 'completed', 'failed', 'skipped')),
    progress_percentage INTEGER DEFAULT 0 CHECK (progress_percentage >= 0 AND progress_percentage <= 100),
    error_message TEXT,
    result_data JSONB, -- Stage-specific results
    started_at TIMESTAMP WITH TIME ZONE,
    completed_at TIMESTAMP WITH TIME ZONE,
    retry_count INTEGER DEFAULT 0,
    scene_index INTEGER, -- For scene_creation and consistency_validation stages (0-based index)
    UNIQUE(job_id, stage_name, scene_index) -- Ensure unique stage per scene
);

-- Create indexes for better performance
CREATE INDEX IF NOT EXISTS idx_book_generation_jobs_status ON book_generation_jobs(status);
CREATE INDEX IF NOT EXISTS idx_book_generation_jobs_priority ON book_generation_jobs(priority);
CREATE INDEX IF NOT EXISTS idx_book_generation_jobs_created_at ON book_generation_jobs(created_at);
CREATE INDEX IF NOT EXISTS idx_book_generation_jobs_user_id ON book_generation_jobs(user_id);
CREATE INDEX IF NOT EXISTS idx_book_generation_jobs_book_id ON book_generation_jobs(book_id);
CREATE INDEX IF NOT EXISTS idx_job_stages_job_id ON job_stages(job_id);
CREATE INDEX IF NOT EXISTS idx_job_stages_status ON job_stages(status);
CREATE INDEX IF NOT EXISTS idx_job_stages_stage_name ON job_stages(stage_name);

-- Enable Row Level Security (RLS)
ALTER TABLE book_generation_jobs ENABLE ROW LEVEL SECURITY;
ALTER TABLE job_stages ENABLE ROW LEVEL SECURITY;

-- RLS Policies for book_generation_jobs
CREATE POLICY "Users can view own jobs" ON book_generation_jobs
    FOR SELECT USING (auth.uid() = user_id);

CREATE POLICY "Users can insert own jobs" ON book_generation_jobs
    FOR INSERT WITH CHECK (auth.uid() = user_id);

CREATE POLICY "Users can update own jobs" ON book_generation_jobs
    FOR UPDATE USING (auth.uid() = user_id);

CREATE POLICY "Users can delete own jobs" ON book_generation_jobs
    FOR DELETE USING (auth.uid() = user_id);

-- RLS Policies for job_stages (users can access stages for their own jobs)
CREATE POLICY "Users can view own job stages" ON job_stages
    FOR SELECT USING (
        job_id IN (SELECT id FROM book_generation_jobs WHERE user_id = auth.uid())
    );

CREATE POLICY "Users can insert own job stages" ON job_stages
    FOR INSERT WITH CHECK (
        job_id IN (SELECT id FROM book_generation_jobs WHERE user_id = auth.uid())
    );

CREATE POLICY "Users can update own job stages" ON job_stages
    FOR UPDATE USING (
        job_id IN (SELECT id FROM book_generation_jobs WHERE user_id = auth.uid())
    );

CREATE POLICY "Users can delete own job stages" ON job_stages
    FOR DELETE USING (
        job_id IN (SELECT id FROM book_generation_jobs WHERE user_id = auth.uid())
    );

-- Function to update updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Triggers to auto-update updated_at
CREATE TRIGGER update_book_generation_jobs_updated_at BEFORE UPDATE ON book_generation_jobs
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_job_stages_updated_at BEFORE UPDATE ON job_stages
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Grant permissions
GRANT ALL ON book_generation_jobs TO authenticated;
GRANT ALL ON job_stages TO authenticated;
GRANT USAGE, SELECT ON SEQUENCE book_generation_jobs_id_seq TO authenticated;
GRANT USAGE, SELECT ON SEQUENCE job_stages_id_seq TO authenticated;


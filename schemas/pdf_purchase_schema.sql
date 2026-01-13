-- PDF Purchase Verification Schema
-- Run this SQL in your Supabase SQL Editor

-- Add pdf_url column to stories table if it doesn't exist
ALTER TABLE stories 
ADD COLUMN IF NOT EXISTS pdf_url TEXT;

-- Create book_purchases table for purchase verification
CREATE TABLE IF NOT EXISTS book_purchases (
    id BIGSERIAL PRIMARY KEY,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    story_id BIGINT NOT NULL REFERENCES stories(id) ON DELETE CASCADE,
    user_id UUID NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
    purchase_status VARCHAR DEFAULT 'completed' CHECK (purchase_status IN ('pending', 'completed', 'failed', 'refunded')),
    purchase_date TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    transaction_id TEXT, -- External payment transaction ID (if applicable)
    amount_paid DECIMAL(10, 2), -- Amount paid (if applicable)
    payment_method VARCHAR, -- Payment method (e.g., 'stripe', 'paypal', 'free')
    UNIQUE(story_id, user_id) -- One purchase per user per story
);

-- Create indexes for better performance
CREATE INDEX IF NOT EXISTS idx_book_purchases_story_id ON book_purchases(story_id);
CREATE INDEX IF NOT EXISTS idx_book_purchases_user_id ON book_purchases(user_id);
CREATE INDEX IF NOT EXISTS idx_book_purchases_status ON book_purchases(purchase_status);
CREATE INDEX IF NOT EXISTS idx_stories_pdf_url ON stories(pdf_url) WHERE pdf_url IS NOT NULL;

-- Enable Row Level Security (RLS)
ALTER TABLE book_purchases ENABLE ROW LEVEL SECURITY;

-- Create policies for RLS
-- Policy: Users can only view their own purchases
CREATE POLICY "Users can view own purchases" ON book_purchases
    FOR SELECT USING (user_id = auth.uid());

-- Policy: Users can only insert their own purchases
CREATE POLICY "Users can insert own purchases" ON book_purchases
    FOR INSERT WITH CHECK (user_id = auth.uid());

-- Policy: Service role can do everything (for backend operations)
CREATE POLICY "Service role full access" ON book_purchases
    FOR ALL USING (auth.role() = 'service_role');


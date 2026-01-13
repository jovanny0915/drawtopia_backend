-- Migration for Scheduled Gift Delivery System
-- This migration adds necessary indexes and ensures the schema is correct

-- Ensure gifts table has all required columns
-- (These should already exist, but this ensures they're present)
DO $$ 
BEGIN
    -- Add notification_sent column if it doesn't exist
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns 
        WHERE table_name = 'gifts' AND column_name = 'notification_sent'
    ) THEN
        ALTER TABLE gifts ADD COLUMN notification_sent BOOLEAN DEFAULT FALSE;
    END IF;

    -- Add notification_sent_at column if it doesn't exist
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns 
        WHERE table_name = 'gifts' AND column_name = 'notification_sent_at'
    ) THEN
        ALTER TABLE gifts ADD COLUMN notification_sent_at TIMESTAMP WITH TIME ZONE;
    END IF;

    -- Add checked column if it doesn't exist
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns 
        WHERE table_name = 'gifts' AND column_name = 'checked'
    ) THEN
        ALTER TABLE gifts ADD COLUMN checked BOOLEAN DEFAULT FALSE;
    END IF;

    -- Add to_user_id column if it doesn't exist
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns 
        WHERE table_name = 'gifts' AND column_name = 'to_user_id'
    ) THEN
        ALTER TABLE gifts ADD COLUMN to_user_id UUID REFERENCES auth.users(id);
    END IF;

    -- Add from_user_id column if it doesn't exist
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns 
        WHERE table_name = 'gifts' AND column_name = 'from_user_id'
    ) THEN
        ALTER TABLE gifts ADD COLUMN from_user_id UUID REFERENCES auth.users(id);
    END IF;
END $$;

-- Create index for scheduled delivery queries (critical for performance)
-- This index is used by the cron job to quickly find gifts ready for delivery
DROP INDEX IF EXISTS idx_gifts_scheduled_delivery;
CREATE INDEX idx_gifts_scheduled_delivery 
ON gifts(delivery_time, notification_sent, status, to_user_id) 
WHERE notification_sent = FALSE AND status = 'completed' AND to_user_id IS NOT NULL;

-- Create index for checking notification status
DROP INDEX IF EXISTS idx_gifts_notification_status;
CREATE INDEX idx_gifts_notification_status 
ON gifts(notification_sent, status) 
WHERE notification_sent = FALSE;

-- Create index for user gift lookups
DROP INDEX IF EXISTS idx_gifts_to_user_id;
CREATE INDEX idx_gifts_to_user_id ON gifts(to_user_id) WHERE to_user_id IS NOT NULL;

-- Create index for sender gift lookups
DROP INDEX IF EXISTS idx_gifts_from_user_id;
CREATE INDEX idx_gifts_from_user_id ON gifts(from_user_id) WHERE from_user_id IS NOT NULL;

-- Ensure push_subscriptions table exists
CREATE TABLE IF NOT EXISTS push_subscriptions (
  id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  user_id UUID REFERENCES auth.users(id) NOT NULL,
  endpoint TEXT NOT NULL,
  p256dh TEXT NOT NULL,
  auth TEXT NOT NULL,
  user_agent TEXT,
  created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
  updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
  UNIQUE(user_id, endpoint)
);

-- Create indexes for push_subscriptions
DROP INDEX IF EXISTS idx_push_subscriptions_user_id;
CREATE INDEX idx_push_subscriptions_user_id ON push_subscriptions(user_id);

DROP INDEX IF EXISTS idx_push_subscriptions_endpoint;
CREATE INDEX idx_push_subscriptions_endpoint ON push_subscriptions(endpoint);

-- Add RLS policies for push_subscriptions
ALTER TABLE push_subscriptions ENABLE ROW LEVEL SECURITY;

-- Users can only read their own subscriptions
DROP POLICY IF EXISTS "Users can view own push subscriptions" ON push_subscriptions;
CREATE POLICY "Users can view own push subscriptions" 
ON push_subscriptions FOR SELECT 
USING (auth.uid() = user_id);

-- Users can insert their own subscriptions
DROP POLICY IF EXISTS "Users can insert own push subscriptions" ON push_subscriptions;
CREATE POLICY "Users can insert own push subscriptions" 
ON push_subscriptions FOR INSERT 
WITH CHECK (auth.uid() = user_id);

-- Users can update their own subscriptions
DROP POLICY IF EXISTS "Users can update own push subscriptions" ON push_subscriptions;
CREATE POLICY "Users can update own push subscriptions" 
ON push_subscriptions FOR UPDATE 
USING (auth.uid() = user_id);

-- Users can delete their own subscriptions
DROP POLICY IF EXISTS "Users can delete own push subscriptions" ON push_subscriptions;
CREATE POLICY "Users can delete own push subscriptions" 
ON push_subscriptions FOR DELETE 
USING (auth.uid() = user_id);

-- Service role can access all subscriptions (for edge functions)
DROP POLICY IF EXISTS "Service role can access all push subscriptions" ON push_subscriptions;
CREATE POLICY "Service role can access all push subscriptions" 
ON push_subscriptions FOR ALL 
USING (auth.jwt()->>'role' = 'service_role');

-- Update RLS policies for gifts table to support scheduled delivery
ALTER TABLE gifts ENABLE ROW LEVEL SECURITY;

-- Users can view gifts they sent
DROP POLICY IF EXISTS "Users can view gifts they sent" ON gifts;
CREATE POLICY "Users can view gifts they sent" 
ON gifts FOR SELECT 
USING (auth.uid() = from_user_id OR auth.uid() = user_id);

-- Users can view gifts they received
DROP POLICY IF EXISTS "Users can view gifts they received" ON gifts;
CREATE POLICY "Users can view gifts they received" 
ON gifts FOR SELECT 
USING (auth.uid() = to_user_id);

-- Users can insert gifts they are sending
DROP POLICY IF EXISTS "Users can insert gifts they send" ON gifts;
CREATE POLICY "Users can insert gifts they send" 
ON gifts FOR INSERT 
WITH CHECK (auth.uid() = from_user_id OR auth.uid() = user_id);

-- Users can update gifts they sent (only certain fields)
DROP POLICY IF EXISTS "Users can update gifts they sent" ON gifts;
CREATE POLICY "Users can update gifts they sent" 
ON gifts FOR UPDATE 
USING (auth.uid() = from_user_id OR auth.uid() = user_id);

-- Recipients can mark gifts as checked
DROP POLICY IF EXISTS "Recipients can mark gifts as checked" ON gifts;
CREATE POLICY "Recipients can mark gifts as checked" 
ON gifts FOR UPDATE 
USING (auth.uid() = to_user_id);

-- Service role can access all gifts (for edge functions and cron jobs)
DROP POLICY IF EXISTS "Service role can access all gifts" ON gifts;
CREATE POLICY "Service role can access all gifts" 
ON gifts FOR ALL 
USING (auth.jwt()->>'role' = 'service_role');

-- Create a function to automatically update updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Create trigger for push_subscriptions updated_at
DROP TRIGGER IF EXISTS update_push_subscriptions_updated_at ON push_subscriptions;
CREATE TRIGGER update_push_subscriptions_updated_at
    BEFORE UPDATE ON push_subscriptions
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

-- Add comments for documentation
COMMENT ON TABLE push_subscriptions IS 'Stores web push notification subscriptions for users';
COMMENT ON COLUMN push_subscriptions.endpoint IS 'Push service endpoint URL';
COMMENT ON COLUMN push_subscriptions.p256dh IS 'P256DH key for encryption';
COMMENT ON COLUMN push_subscriptions.auth IS 'Auth secret for encryption';

COMMENT ON COLUMN gifts.notification_sent IS 'Whether push notification has been sent for this gift';
COMMENT ON COLUMN gifts.notification_sent_at IS 'Timestamp when push notification was sent';
COMMENT ON COLUMN gifts.checked IS 'Whether recipient has viewed the gift notification';
COMMENT ON COLUMN gifts.to_user_id IS 'User ID of the gift recipient';
COMMENT ON COLUMN gifts.from_user_id IS 'User ID of the gift sender';

-- Grant necessary permissions
GRANT ALL ON push_subscriptions TO authenticated;
GRANT ALL ON push_subscriptions TO service_role;

GRANT ALL ON gifts TO authenticated;
GRANT ALL ON gifts TO service_role;

-- Create a view for monitoring scheduled deliveries (useful for debugging)
CREATE OR REPLACE VIEW scheduled_gifts_pending AS
SELECT 
    id,
    from_user_id,
    to_user_id,
    child_name,
    occasion,
    delivery_time,
    status,
    notification_sent,
    notification_sent_at,
    created_at,
    EXTRACT(EPOCH FROM (delivery_time - NOW())) / 60 AS minutes_until_delivery
FROM gifts
WHERE 
    notification_sent = FALSE 
    AND status = 'completed'
    AND to_user_id IS NOT NULL
    AND delivery_time > NOW()
ORDER BY delivery_time ASC;

COMMENT ON VIEW scheduled_gifts_pending IS 'View of gifts pending scheduled delivery';

-- Grant access to the view
GRANT SELECT ON scheduled_gifts_pending TO authenticated;
GRANT SELECT ON scheduled_gifts_pending TO service_role;

-- Success message
DO $$ 
BEGIN
    RAISE NOTICE '✅ Scheduled delivery migration completed successfully!';
    RAISE NOTICE '📊 Check scheduled_gifts_pending view for pending deliveries';
END $$;


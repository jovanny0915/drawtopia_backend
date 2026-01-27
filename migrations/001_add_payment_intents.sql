-- Create payment_intents table to track Payment Intent lifecycle
CREATE TABLE IF NOT EXISTS payment_intents (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    payment_intent_id TEXT NOT NULL UNIQUE,
    user_id UUID REFERENCES auth.users(id) ON DELETE SET NULL,
    purchase_type TEXT NOT NULL CHECK (purchase_type IN ('gift', 'single_story', 'story_bundle')),
    gift_id UUID,
    story_id UUID,
    amount INTEGER NOT NULL,
    currency TEXT NOT NULL DEFAULT 'usd',
    status TEXT NOT NULL DEFAULT 'created' CHECK (status IN ('created', 'processing', 'succeeded', 'failed', 'cancelled')),
    error_message TEXT,
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create index for faster lookups
CREATE INDEX IF NOT EXISTS idx_payment_intents_user_id ON payment_intents(user_id);
CREATE INDEX IF NOT EXISTS idx_payment_intents_payment_intent_id ON payment_intents(payment_intent_id);
CREATE INDEX IF NOT EXISTS idx_payment_intents_status ON payment_intents(status);
CREATE INDEX IF NOT EXISTS idx_payment_intents_gift_id ON payment_intents(gift_id);

-- Create transactions table to track completed payments
CREATE TABLE IF NOT EXISTS transactions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    payment_intent_id TEXT NOT NULL,
    user_id UUID REFERENCES auth.users(id) ON DELETE SET NULL,
    purchase_type TEXT NOT NULL CHECK (purchase_type IN ('gift', 'single_story', 'story_bundle', 'subscription')),
    amount INTEGER NOT NULL,
    currency TEXT NOT NULL DEFAULT 'usd',
    status TEXT NOT NULL DEFAULT 'succeeded' CHECK (status IN ('succeeded', 'failed', 'refunded')),
    gift_id UUID,
    story_id UUID,
    subscription_id TEXT,
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create indexes for transactions
CREATE INDEX IF NOT EXISTS idx_transactions_user_id ON transactions(user_id);
CREATE INDEX IF NOT EXISTS idx_transactions_payment_intent_id ON transactions(payment_intent_id);
CREATE INDEX IF NOT EXISTS idx_transactions_gift_id ON transactions(gift_id);
CREATE INDEX IF NOT EXISTS idx_transactions_created_at ON transactions(created_at DESC);

-- Add payment_intent_id to gifts table if it doesn't exist
DO $$ 
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns 
        WHERE table_name = 'gifts' AND column_name = 'payment_intent_id'
    ) THEN
        ALTER TABLE gifts ADD COLUMN payment_intent_id TEXT;
        CREATE INDEX IF NOT EXISTS idx_gifts_payment_intent_id ON gifts(payment_intent_id);
    END IF;
END $$;

-- Add payment_status to gifts table if it doesn't exist
DO $$ 
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns 
        WHERE table_name = 'gifts' AND column_name = 'payment_status'
    ) THEN
        ALTER TABLE gifts ADD COLUMN payment_status TEXT DEFAULT 'pending' CHECK (payment_status IN ('pending', 'paid', 'failed', 'refunded'));
    END IF;
END $$;

-- Create function to automatically update updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Create trigger for payment_intents table
DROP TRIGGER IF EXISTS update_payment_intents_updated_at ON payment_intents;
CREATE TRIGGER update_payment_intents_updated_at
    BEFORE UPDATE ON payment_intents
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

-- Enable Row Level Security (RLS)
ALTER TABLE payment_intents ENABLE ROW LEVEL SECURITY;
ALTER TABLE transactions ENABLE ROW LEVEL SECURITY;

-- Create policies for payment_intents
CREATE POLICY "Users can view their own payment intents"
    ON payment_intents FOR SELECT
    USING (auth.uid() = user_id);

CREATE POLICY "Service role can manage all payment intents"
    ON payment_intents FOR ALL
    USING (auth.role() = 'service_role');

-- Create policies for transactions
CREATE POLICY "Users can view their own transactions"
    ON transactions FOR SELECT
    USING (auth.uid() = user_id);

CREATE POLICY "Service role can manage all transactions"
    ON transactions FOR ALL
    USING (auth.role() = 'service_role');

-- Grant necessary permissions
GRANT SELECT, INSERT, UPDATE ON payment_intents TO authenticated;
GRANT SELECT ON transactions TO authenticated;
GRANT ALL ON payment_intents TO service_role;
GRANT ALL ON transactions TO service_role;

-- Add comments for documentation
COMMENT ON TABLE payment_intents IS 'Tracks Stripe Payment Intent lifecycle for all purchases';
COMMENT ON TABLE transactions IS 'Records completed payment transactions';
COMMENT ON COLUMN payment_intents.payment_intent_id IS 'Stripe Payment Intent ID';
COMMENT ON COLUMN payment_intents.status IS 'Current status of the payment intent';
COMMENT ON COLUMN transactions.payment_intent_id IS 'Reference to the Stripe Payment Intent that created this transaction';

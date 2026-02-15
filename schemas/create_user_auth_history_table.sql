-- Stores authentication history events for analytics (login/register only)
CREATE TABLE IF NOT EXISTS user_auth_history (
    id BIGSERIAL PRIMARY KEY,
    user_id UUID REFERENCES auth.users(id) ON DELETE SET NULL,
    event_type TEXT NOT NULL CHECK (event_type IN ('login', 'register')),
    auth_provider TEXT NOT NULL DEFAULT 'password',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_user_auth_history_created_at
    ON user_auth_history(created_at DESC);

CREATE INDEX IF NOT EXISTS idx_user_auth_history_event_type_created_at
    ON user_auth_history(event_type, created_at DESC);

ALTER TABLE user_auth_history ENABLE ROW LEVEL SECURITY;

DROP POLICY IF EXISTS "Service role full access on user_auth_history" ON user_auth_history;
CREATE POLICY "Service role full access on user_auth_history"
    ON user_auth_history
    FOR ALL
    USING (auth.role() = 'service_role')
    WITH CHECK (auth.role() = 'service_role');

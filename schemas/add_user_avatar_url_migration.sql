-- Add avatar_url to users table for Google OAuth profile photo
ALTER TABLE users
ADD COLUMN IF NOT EXISTS avatar_url TEXT;

-- Optional index to speed up filters that check for avatar existence
CREATE INDEX IF NOT EXISTS idx_users_avatar_url_not_null
ON users (id)
WHERE avatar_url IS NOT NULL;

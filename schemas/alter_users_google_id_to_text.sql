-- Google OAuth provider IDs can exceed BIGINT range, so store them as text.
ALTER TABLE users
ALTER COLUMN google_id TYPE TEXT
USING google_id::TEXT;

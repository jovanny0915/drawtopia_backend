-- Create/repair the Supabase Storage bucket used by child profile avatars.
--
-- The frontend uploads to:
--   avatars/<auth.uid()>/<filename>
--
-- These policies allow public reads for displayed avatars, while writes are
-- restricted to the authenticated user's own folder.

INSERT INTO storage.buckets (
    id,
    name,
    public,
    file_size_limit,
    allowed_mime_types
)
VALUES (
    'avatars',
    'avatars',
    true,
    5242880,
    ARRAY['image/jpeg', 'image/jpg', 'image/png', 'image/webp']::text[]
)
ON CONFLICT (id) DO UPDATE
SET
    public = EXCLUDED.public,
    file_size_limit = EXCLUDED.file_size_limit,
    allowed_mime_types = EXCLUDED.allowed_mime_types;

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1
        FROM pg_policies
        WHERE schemaname = 'storage'
          AND tablename = 'objects'
          AND policyname = 'Avatar images are publicly readable'
    ) THEN
        CREATE POLICY "Avatar images are publicly readable"
        ON storage.objects
        FOR SELECT
        TO public
        USING (bucket_id = 'avatars');
    END IF;

    IF NOT EXISTS (
        SELECT 1
        FROM pg_policies
        WHERE schemaname = 'storage'
          AND tablename = 'objects'
          AND policyname = 'Users can upload avatars to their own folder'
    ) THEN
        CREATE POLICY "Users can upload avatars to their own folder"
        ON storage.objects
        FOR INSERT
        TO authenticated
        WITH CHECK (
            bucket_id = 'avatars'
            AND (
                (storage.foldername(name))[1] = auth.uid()::text
                OR name LIKE auth.uid()::text || '\_%' ESCAPE '\'
            )
        );
    END IF;

    IF NOT EXISTS (
        SELECT 1
        FROM pg_policies
        WHERE schemaname = 'storage'
          AND tablename = 'objects'
          AND policyname = 'Users can update avatars in their own folder'
    ) THEN
        CREATE POLICY "Users can update avatars in their own folder"
        ON storage.objects
        FOR UPDATE
        TO authenticated
        USING (
            bucket_id = 'avatars'
            AND (
                (storage.foldername(name))[1] = auth.uid()::text
                OR name LIKE auth.uid()::text || '\_%' ESCAPE '\'
            )
        )
        WITH CHECK (
            bucket_id = 'avatars'
            AND (
                (storage.foldername(name))[1] = auth.uid()::text
                OR name LIKE auth.uid()::text || '\_%' ESCAPE '\'
            )
        );
    END IF;

    IF NOT EXISTS (
        SELECT 1
        FROM pg_policies
        WHERE schemaname = 'storage'
          AND tablename = 'objects'
          AND policyname = 'Users can delete avatars from their own folder'
    ) THEN
        CREATE POLICY "Users can delete avatars from their own folder"
        ON storage.objects
        FOR DELETE
        TO authenticated
        USING (
            bucket_id = 'avatars'
            AND (
                (storage.foldername(name))[1] = auth.uid()::text
                OR name LIKE auth.uid()::text || '\_%' ESCAPE '\'
            )
        );
    END IF;
END $$;

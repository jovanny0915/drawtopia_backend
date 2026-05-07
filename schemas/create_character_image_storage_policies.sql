-- Create/repair Supabase Storage policies for user-uploaded character images.
--
-- The frontend uploads character reference images to:
--   images/<auth.uid()>/character_<timestamp>_<random>.<ext>
--
-- Backend-generated images also use the `images` bucket through the service
-- role key, so these policies only grant direct browser writes for
-- authenticated users inside their own folder.

INSERT INTO storage.buckets (id, name, public)
VALUES ('images', 'images', true)
ON CONFLICT (id) DO UPDATE
SET public = EXCLUDED.public;

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1
        FROM pg_policies
        WHERE schemaname = 'storage'
          AND tablename = 'objects'
          AND policyname = 'Images are publicly readable'
    ) THEN
        CREATE POLICY "Images are publicly readable"
        ON storage.objects
        FOR SELECT
        TO public
        USING (bucket_id = 'images');
    END IF;

    IF NOT EXISTS (
        SELECT 1
        FROM pg_policies
        WHERE schemaname = 'storage'
          AND tablename = 'objects'
          AND policyname = 'Users can upload character images to their own folder'
    ) THEN
        CREATE POLICY "Users can upload character images to their own folder"
        ON storage.objects
        FOR INSERT
        TO authenticated
        WITH CHECK (
            bucket_id = 'images'
            AND (
                (storage.foldername(name))[1] = auth.uid()::text
                OR name LIKE 'character\_' || auth.uid()::text || '\_%' ESCAPE '\'
            )
        );
    END IF;

    IF NOT EXISTS (
        SELECT 1
        FROM pg_policies
        WHERE schemaname = 'storage'
          AND tablename = 'objects'
          AND policyname = 'Users can update character images in their own folder'
    ) THEN
        CREATE POLICY "Users can update character images in their own folder"
        ON storage.objects
        FOR UPDATE
        TO authenticated
        USING (
            bucket_id = 'images'
            AND (
                (storage.foldername(name))[1] = auth.uid()::text
                OR name LIKE 'character\_' || auth.uid()::text || '\_%' ESCAPE '\'
            )
        )
        WITH CHECK (
            bucket_id = 'images'
            AND (
                (storage.foldername(name))[1] = auth.uid()::text
                OR name LIKE 'character\_' || auth.uid()::text || '\_%' ESCAPE '\'
            )
        );
    END IF;

    IF NOT EXISTS (
        SELECT 1
        FROM pg_policies
        WHERE schemaname = 'storage'
          AND tablename = 'objects'
          AND policyname = 'Users can delete character images from their own folder'
    ) THEN
        CREATE POLICY "Users can delete character images from their own folder"
        ON storage.objects
        FOR DELETE
        TO authenticated
        USING (
            bucket_id = 'images'
            AND (
                (storage.foldername(name))[1] = auth.uid()::text
                OR name LIKE 'character\_' || auth.uid()::text || '\_%' ESCAPE '\'
            )
        );
    END IF;
END $$;

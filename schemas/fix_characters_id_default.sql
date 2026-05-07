-- Ensure characters.id, uid, and timestamps are auto-populated.
--
-- Some environments created the `characters` table with a NOT NULL id column
-- but without a default sequence, which causes inserts from the frontend to
-- fail with:
--   "null value in column \"id\" of relation \"characters\"
--    violates not-null constraint"
--
-- The frontend intentionally omits `id` when creating characters. The database
-- should generate it.

CREATE EXTENSION IF NOT EXISTS pgcrypto;

DO $$
DECLARE
    column_type TEXT;
    max_existing_id BIGINT;
BEGIN
    SELECT data_type
    INTO column_type
    FROM information_schema.columns
    WHERE table_schema = 'public'
      AND table_name = 'characters'
      AND column_name = 'id';

    IF column_type IS NULL THEN
        RAISE EXCEPTION 'Column public.characters.id does not exist';
    END IF;

    IF column_type NOT IN ('bigint', 'integer', 'smallint') THEN
        RAISE EXCEPTION 'public.characters.id must be numeric to use this migration, found %', column_type;
    END IF;

    CREATE SEQUENCE IF NOT EXISTS public.characters_id_seq;

    ALTER SEQUENCE public.characters_id_seq
        OWNED BY public.characters.id;

    SELECT COALESCE(MAX(id), 0)
    INTO max_existing_id
    FROM public.characters;

    IF max_existing_id > 0 THEN
        PERFORM setval('public.characters_id_seq'::regclass, max_existing_id, true);
    ELSE
        -- Empty table: initialize so the first generated id is 1.
        PERFORM setval('public.characters_id_seq'::regclass, 1, false);
    END IF;

    ALTER TABLE public.characters
        ALTER COLUMN id SET DEFAULT nextval('public.characters_id_seq'::regclass);

    ALTER TABLE public.characters
        ALTER COLUMN id SET NOT NULL;
END $$;

DO $$
DECLARE
    uid_column_type TEXT;
BEGIN
    SELECT data_type
    INTO uid_column_type
    FROM information_schema.columns
    WHERE table_schema = 'public'
      AND table_name = 'characters'
      AND column_name = 'uid';

    IF uid_column_type IS NULL THEN
        -- Some schemas only use numeric id for characters.
        RETURN;
    END IF;

    IF uid_column_type = 'uuid' THEN
        UPDATE public.characters
        SET uid = gen_random_uuid()
        WHERE uid IS NULL;

        ALTER TABLE public.characters
            ALTER COLUMN uid SET DEFAULT gen_random_uuid();
    ELSIF uid_column_type IN ('text', 'character varying', 'character') THEN
        UPDATE public.characters
        SET uid = gen_random_uuid()::text
        WHERE uid IS NULL;

        ALTER TABLE public.characters
            ALTER COLUMN uid SET DEFAULT gen_random_uuid()::text;
    ELSE
        RAISE EXCEPTION 'public.characters.uid must be uuid or text-like to use this migration, found %', uid_column_type;
    END IF;

    ALTER TABLE public.characters
        ALTER COLUMN uid SET NOT NULL;
END $$;

ALTER TABLE public.characters
    ALTER COLUMN created_at SET DEFAULT NOW(),
    ALTER COLUMN updated_at SET DEFAULT NOW();

UPDATE public.characters
SET
    created_at = COALESCE(created_at, NOW()),
    updated_at = COALESCE(updated_at, NOW())
WHERE created_at IS NULL
   OR updated_at IS NULL;

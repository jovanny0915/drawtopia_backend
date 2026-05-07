-- Ensure child_profiles.id is auto-populated.
--
-- Some environments created the `child_profiles` table with a NOT NULL id
-- column but without a default sequence, which causes inserts from the
-- frontend to fail with:
--   "null value in column \"id\" of relation \"child_profiles\"
--    violates not-null constraint"
--
-- The frontend intentionally omits `id` when creating child profiles. The
-- database should generate it.

DO $$
DECLARE
    column_type TEXT;
    max_existing_id BIGINT;
BEGIN
    SELECT data_type
    INTO column_type
    FROM information_schema.columns
    WHERE table_schema = 'public'
      AND table_name = 'child_profiles'
      AND column_name = 'id';

    IF column_type IS NULL THEN
        RAISE EXCEPTION 'Column public.child_profiles.id does not exist';
    END IF;

    IF column_type NOT IN ('bigint', 'integer', 'smallint') THEN
        RAISE EXCEPTION 'public.child_profiles.id must be numeric to use this migration, found %', column_type;
    END IF;

    CREATE SEQUENCE IF NOT EXISTS public.child_profiles_id_seq;

    ALTER SEQUENCE public.child_profiles_id_seq
        OWNED BY public.child_profiles.id;

    SELECT COALESCE(MAX(id), 0)
    INTO max_existing_id
    FROM public.child_profiles;

    IF max_existing_id > 0 THEN
        PERFORM setval('public.child_profiles_id_seq'::regclass, max_existing_id, true);
    ELSE
        -- Empty table: initialize so the first generated id is 1.
        PERFORM setval('public.child_profiles_id_seq'::regclass, 1, false);
    END IF;

    ALTER TABLE public.child_profiles
        ALTER COLUMN id SET DEFAULT nextval('public.child_profiles_id_seq'::regclass);

    ALTER TABLE public.child_profiles
        ALTER COLUMN id SET NOT NULL;
END $$;

-- Run this SQL in your Supabase SQL Editor after create_ai_prompt_documents_table.sql.
-- Adds editable gender/body/clothing prompts to the prompt_image document.

WITH defaults AS (
  SELECT
    'Legacy fallback only: do not copy the template character''s clothing, costume, body type, gender presentation, or accessories. Preserve the reference character identity and use the gender appearance prompt when available.'::text AS legacy_outfit_prompt,
    jsonb_build_object(
      'male',
      'GENDER AND CLOTHING RULE (MALE CHARACTER): The main character must read as the same male boy/man from the reference character image. Preserve the reference character''s face, hair, skin tone, body proportions, and age impression. Do not copy the template character''s clothing, costume, body shape, gender presentation, hairstyle, or accessories. Dress the character in boy-appropriate child-friendly adventure clothing that matches the story world and page action while staying consistent across pages.',
      'female',
      'GENDER AND CLOTHING RULE (FEMALE CHARACTER): The main character must read as the same female girl/woman from the reference character image. Preserve the reference character''s face, hair, skin tone, body proportions, and age impression. Do not copy the template character''s clothing, costume, body shape, gender presentation, hairstyle, or accessories. Dress the character in girl-appropriate child-friendly adventure clothing that matches the story world and page action while staying consistent across pages.',
      'neutral',
      'GENDER AND CLOTHING RULE (NEUTRAL OR UNSPECIFIED CHARACTER): Preserve the reference character''s face, hair, skin tone, body proportions, age impression, and neutral gender presentation. Do not copy the template character''s clothing, costume, body shape, gender presentation, hairstyle, or accessories. Dress the character in child-friendly adventure clothing that matches the story world and page action while staying consistent across pages.'
    ) AS gender_prompts
)
UPDATE ai_prompt_documents
SET
  content = jsonb_set(
    jsonb_set(
      content,
      '{personTemplateOutfitPrompt}',
      to_jsonb(defaults.legacy_outfit_prompt),
      true
    ),
    '{personGenderAppearancePrompts}',
    defaults.gender_prompts || COALESCE(content->'personGenderAppearancePrompts', '{}'::jsonb),
    true
  ),
  version = version + 1,
  updated_at = NOW()
FROM defaults
WHERE file_key = 'prompt_image';

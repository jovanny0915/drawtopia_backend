-- Run this SQL in your Supabase SQL Editor.
-- Stores editable AI prompt JSON documents used by the admin prompt manager.

CREATE TABLE IF NOT EXISTS ai_prompt_documents (
  file_key TEXT PRIMARY KEY,
  content JSONB NOT NULL,
  description TEXT,
  version INTEGER NOT NULL DEFAULT 1,
  updated_by UUID,
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_ai_prompt_documents_updated_at
  ON ai_prompt_documents(updated_at DESC);

COMMENT ON TABLE ai_prompt_documents IS 'Editable AI prompt JSON documents for generation flows.';
COMMENT ON COLUMN ai_prompt_documents.file_key IS 'Stable prompt document key, e.g. prompt1, prompt_image, prompt_story, backend_prompts.';
COMMENT ON COLUMN ai_prompt_documents.content IS 'Full JSON document used as runtime prompt configuration.';
COMMENT ON COLUMN ai_prompt_documents.version IS 'Incremented every time the prompt document is saved.';

ALTER TABLE ai_prompt_documents ENABLE ROW LEVEL SECURITY;

DROP POLICY IF EXISTS "Service role can manage AI prompt documents" ON ai_prompt_documents;
CREATE POLICY "Service role can manage AI prompt documents"
  ON ai_prompt_documents
  FOR ALL
  USING (auth.role() = 'service_role')
  WITH CHECK (auth.role() = 'service_role');

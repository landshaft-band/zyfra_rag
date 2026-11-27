CREATE TABLE conversations (
    id uuid PRIMARY KEY,
    owner_digest text NOT NULL,
    created_at timestamptz NOT NULL DEFAULT now(),
    next_position bigint NOT NULL DEFAULT 1 CHECK (next_position > 0)
);

CREATE INDEX conversations_owner_created_idx
    ON conversations (owner_digest, created_at DESC, id DESC);

CREATE TABLE messages (
    conversation_id uuid NOT NULL REFERENCES conversations (id) ON DELETE CASCADE,
    position bigint NOT NULL CHECK (position > 0),
    role text NOT NULL CHECK (role IN ('user', 'assistant')),
    content text NOT NULL,
    result jsonb CHECK (result IS NULL OR (role = 'assistant' AND jsonb_typeof(result) = 'object')),
    created_at timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (conversation_id, position)
);

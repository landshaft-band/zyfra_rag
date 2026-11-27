CREATE TABLE jobs (
    id uuid PRIMARY KEY,
    conversation_id uuid NOT NULL REFERENCES conversations (id) ON DELETE CASCADE,
    sequence bigint GENERATED ALWAYS AS IDENTITY UNIQUE,
    request_id uuid NOT NULL,
    question text NOT NULL CHECK (length(question) BETWEEN 1 AND 4000),
    status text NOT NULL DEFAULT 'queued' CHECK (status IN ('queued', 'running', 'completed', 'failed')),
    attempts integer NOT NULL DEFAULT 0,
    signal_at timestamptz,
    lease_id uuid,
    leased_until timestamptz,
    assistant_position bigint,
    error text,
    created_at timestamptz NOT NULL DEFAULT now(),
    UNIQUE (conversation_id, request_id),
    FOREIGN KEY (conversation_id, assistant_position) REFERENCES messages (conversation_id, position),
    CHECK ((status = 'completed') = (assistant_position IS NOT NULL)),
    CHECK ((status = 'running') = (lease_id IS NOT NULL AND leased_until IS NOT NULL))
);

CREATE INDEX jobs_pending_idx ON jobs (conversation_id, sequence)
    WHERE status IN ('queued', 'running');
CREATE INDEX jobs_signals_idx ON jobs (signal_at) WHERE status = 'queued';
CREATE INDEX jobs_leases_idx ON jobs (leased_until) WHERE status = 'running';

CREATE TABLE job_outbox (
    id bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    job_id uuid NOT NULL REFERENCES jobs (id) ON DELETE CASCADE,
    kind text NOT NULL CHECK (kind IN ('job.ready', 'job.completed', 'job.failed')),
    available_at timestamptz NOT NULL DEFAULT now(),
    published_at timestamptz
);

CREATE INDEX job_outbox_ready_idx ON job_outbox (available_at, id) WHERE published_at IS NULL;
CREATE INDEX job_outbox_pending_job_idx ON job_outbox (job_id) WHERE published_at IS NULL;

ALTER TABLE jobs ADD COLUMN input_guard_mode text
    CHECK (input_guard_mode IN ('observe', 'enforce'));

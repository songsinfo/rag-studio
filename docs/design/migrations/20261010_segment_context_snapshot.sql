-- Deploy after backing up context_segment / qa_run / job. Additive only.
-- Rollback application without dropping these columns, to retain history.
ALTER TABLE context_segment
  ADD COLUMN context_payload JSON NULL AFTER scope_mode,
  ADD COLUMN context_version INT NOT NULL DEFAULT 0 AFTER context_payload,
  ADD COLUMN context_through_turn INT NOT NULL DEFAULT 0 AFTER context_version,
  ADD COLUMN context_boundary_turn INT NOT NULL DEFAULT 0 AFTER context_through_turn,
  ADD COLUMN context_last_run_id CHAR(36) CHARACTER SET ascii COLLATE ascii_bin NULL AFTER context_boundary_turn;
-- Existing segments are lazily initialized from authorized historic runs when
-- their first new Agent turn succeeds; there is no tenant-wide unsafe backfill.

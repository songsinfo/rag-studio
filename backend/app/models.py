from datetime import datetime, timezone
from sqlalchemy import (
    MetaData, Table, Column, String, DateTime, Boolean, BigInteger, Integer, LargeBinary, Text, JSON, Numeric, CheckConstraint
)
from sqlalchemy.sql import func

metadata = MetaData()

tenant_table = Table(
    "tenant",
    metadata,
    Column("id", String(36), primary_key=True),
    Column("code", String(64), nullable=False, unique=True),
    Column("name", String(128), nullable=False),
    Column("security_epoch", BigInteger, nullable=False, default=0, server_default="0"),
    Column("active_retrieval_config_id", String(36), nullable=True),
    Column("created_at", DateTime(6), nullable=False, default=lambda: datetime.now(timezone.utc), server_default=func.now()),
)

app_user_table = Table(
    "app_user",
    metadata,
    Column("tenant_id", String(36), primary_key=True),
    Column("id", String(36), primary_key=True),
    Column("login_name", String(128), nullable=False),
    Column("password_hash", String(255), nullable=False),
    Column("is_admin", Boolean, nullable=False, default=False, server_default="0"),
    Column("status", String(16), nullable=False, default="ACTIVE", server_default="ACTIVE"),
    Column("created_at", DateTime(6), nullable=False, default=lambda: datetime.now(timezone.utc), server_default=func.now()),
    Column("updated_at", DateTime(6), nullable=False, default=lambda: datetime.now(timezone.utc), server_default=func.now()),
)

auth_session_table = Table(
    "auth_session",
    metadata,
    Column("tenant_id", String(36), primary_key=True),
    Column("id", String(36), primary_key=True),
    Column("user_id", String(36), nullable=False),
    Column("token_hash", LargeBinary(32), nullable=False, unique=True),
    Column("expires_at", DateTime(6), nullable=False),
    Column("revoked_at", DateTime(6), nullable=True),
    Column("created_at", DateTime(6), nullable=False, default=lambda: datetime.now(timezone.utc), server_default=func.now()),
    Column("updated_at", DateTime(6), nullable=False, default=lambda: datetime.now(timezone.utc), server_default=func.now()),
)

user_group_table = Table(
    "user_group",
    metadata,
    Column("tenant_id", String(36), primary_key=True),
    Column("id", String(36), primary_key=True),
    Column("name", String(128), nullable=False),
    Column("created_at", DateTime(6), nullable=False, default=lambda: datetime.now(timezone.utc), server_default=func.now()),
    Column("updated_at", DateTime(6), nullable=False, default=lambda: datetime.now(timezone.utc), server_default=func.now()),
)

group_member_table = Table(
    "group_member",
    metadata,
    Column("tenant_id", String(36), primary_key=True),
    Column("id", String(36), primary_key=True),
    Column("group_id", String(36), nullable=False),
    Column("user_id", String(36), nullable=False),
    Column("created_at", DateTime(6), nullable=False, default=lambda: datetime.now(timezone.utc), server_default=func.now()),
    Column("updated_at", DateTime(6), nullable=False, default=lambda: datetime.now(timezone.utc), server_default=func.now()),
)

knowledge_base_table = Table(
    "knowledge_base",
    metadata,
    Column("tenant_id", String(36), primary_key=True),
    Column("id", String(36), primary_key=True),
    Column("name", String(255), nullable=False),
    Column("status", String(16), nullable=False, default="LIVE", server_default="LIVE"),
    Column("row_version", BigInteger, nullable=False, default=0, server_default="0"),
    Column("created_at", DateTime(6), nullable=False, default=lambda: datetime.now(timezone.utc), server_default=func.now()),
    Column("updated_at", DateTime(6), nullable=False, default=lambda: datetime.now(timezone.utc), server_default=func.now()),
)

kb_grant_table = Table(
    "kb_grant",
    metadata,
    Column("tenant_id", String(36), primary_key=True),
    Column("id", String(36), primary_key=True),
    Column("kb_id", String(36), nullable=False),
    Column("principal_type", String(8), nullable=False),
    Column("user_id", String(36), nullable=True),
    Column("group_id", String(36), nullable=True),
    Column("created_at", DateTime(6), nullable=False, default=lambda: datetime.now(timezone.utc), server_default=func.now()),
    Column("updated_at", DateTime(6), nullable=False, default=lambda: datetime.now(timezone.utc), server_default=func.now()),
)

document_table = Table(
    "document",
    metadata,
    Column("tenant_id", String(36), primary_key=True),
    Column("id", String(36), primary_key=True),
    Column("kb_id", String(36), nullable=False),
    Column("title", String(512), nullable=True),
    Column("acl_mode", String(16), nullable=False, default="RESTRICT", server_default="RESTRICT"),
    Column("status", String(20), nullable=False, default="LIVE", server_default="LIVE"),
    Column("active_build_id", String(36), nullable=True),
    Column("lifecycle_epoch", BigInteger, nullable=False, default=0, server_default="0"),
    Column("row_version", BigInteger, nullable=False, default=0, server_default="0"),
    Column("deleted_at", DateTime(6), nullable=True),
    Column("created_at", DateTime(6), nullable=False, default=lambda: datetime.now(timezone.utc), server_default=func.now()),
    Column("updated_at", DateTime(6), nullable=False, default=lambda: datetime.now(timezone.utc), server_default=func.now()),
)

document_grant_table = Table(
    "document_grant",
    metadata,
    Column("tenant_id", String(36), primary_key=True),
    Column("id", String(36), primary_key=True),
    Column("document_id", String(36), nullable=False),
    Column("principal_type", String(8), nullable=False),
    Column("user_id", String(36), nullable=True),
    Column("group_id", String(36), nullable=True),
    Column("created_at", DateTime(6), nullable=False, default=lambda: datetime.now(timezone.utc), server_default=func.now()),
    Column("updated_at", DateTime(6), nullable=False, default=lambda: datetime.now(timezone.utc), server_default=func.now()),
)

artifact_table = Table(
    "artifact",
    metadata,
    Column("tenant_id", String(36), primary_key=True),
    Column("id", String(36), primary_key=True),
    Column("document_id", String(36), nullable=False),
    Column("kind", String(32), nullable=False),
    Column("storage_key", String(512), nullable=False, unique=True),
    Column("sha256", LargeBinary(32), nullable=False),
    Column("size_bytes", BigInteger, nullable=False),
    Column("status", String(16), nullable=False, default="READY", server_default="READY"),
    Column("created_at", DateTime(6), nullable=False, default=lambda: datetime.now(timezone.utc), server_default=func.now()),
    Column("updated_at", DateTime(6), nullable=False, default=lambda: datetime.now(timezone.utc), server_default=func.now()),
)

document_version_table = Table(
    "document_version",
    metadata,
    Column("tenant_id", String(36), primary_key=True),
    Column("id", String(36), primary_key=True),
    Column("document_id", String(36), nullable=False),
    Column("version_no", Integer, nullable=False),
    Column("source_artifact_id", String(36), nullable=True),
    Column("source_hash", LargeBinary(32), nullable=False),
    Column("original_name", String(512), nullable=True),
    Column("media_type", String(128), nullable=False),
    Column("created_by", String(36), nullable=False),
    Column("created_at", DateTime(6), nullable=False, default=lambda: datetime.now(timezone.utc), server_default=func.now()),
    Column("updated_at", DateTime(6), nullable=False, default=lambda: datetime.now(timezone.utc), server_default=func.now()),
)

config_revision_table = Table(
    "config_revision",
    metadata,
    Column("tenant_id", String(36), primary_key=True),
    Column("id", String(36), primary_key=True),
    Column("kind", String(24), nullable=False),
    Column("name", String(128), nullable=False),
    Column("revision", Integer, nullable=False),
    Column("sha256", LargeBinary(32), nullable=False),
    Column("payload", JSON, nullable=False),
    Column("status", String(16), nullable=False, default="ACTIVE", server_default="ACTIVE"),
    Column("created_at", DateTime(6), nullable=False, default=lambda: datetime.now(timezone.utc), server_default=func.now()),
    Column("updated_at", DateTime(6), nullable=False, default=lambda: datetime.now(timezone.utc), server_default=func.now()),
)

index_build_table = Table(
    "index_build",
    metadata,
    Column("tenant_id", String(36), primary_key=True),
    Column("id", String(36), primary_key=True),
    Column("document_id", String(36), nullable=False),
    Column("version_id", String(36), nullable=False),
    Column("build_no", Integer, nullable=False),
    Column("config_id", String(36), nullable=False),
    Column("status", String(24), nullable=False, default="QUEUED", server_default="QUEUED"),
    Column("model_fingerprint", String(255), nullable=True),
    Column("embedding_dims", Integer, nullable=True),
    Column("capabilities", JSON, nullable=False),
    Column("manifest_hash", LargeBinary(32), nullable=True),
    Column("fence", BigInteger, nullable=False, default=0, server_default="0"),
    Column("published_at", DateTime(6), nullable=True),
    Column("quality_warnings", JSON, nullable=True),
    Column("review_reason", String(2000), nullable=True),
    Column("created_at", DateTime(6), nullable=False, default=lambda: datetime.now(timezone.utc), server_default=func.now()),
    Column("updated_at", DateTime(6), nullable=False, default=lambda: datetime.now(timezone.utc), server_default=func.now()),
)

source_unit_table = Table(
    "source_unit",
    metadata,
    Column("tenant_id", String(36), primary_key=True),
    Column("id", String(36), primary_key=True),
    Column("build_id", String(36), nullable=False),
    Column("document_id", String(36), nullable=False),
    Column("unit_key", String(191), nullable=False),
    Column("kind", String(16), nullable=False),
    Column("text_content", Text, nullable=True),
    Column("location", JSON, nullable=False),
    Column("machine_generated", Boolean, nullable=False, default=False, server_default="0"),
    Column("artifact_id", String(36), nullable=True),
    Column("content_hash", LargeBinary(32), nullable=False),
    Column("created_at", DateTime(6), nullable=False, default=lambda: datetime.now(timezone.utc), server_default=func.now()),
    Column("updated_at", DateTime(6), nullable=False, default=lambda: datetime.now(timezone.utc), server_default=func.now()),
)

chunk_table = Table(
    "chunk",
    metadata,
    Column("tenant_id", String(36), primary_key=True),
    Column("id", String(36), primary_key=True),
    Column("build_id", String(36), nullable=False),
    Column("parent_id", String(36), nullable=True),
    Column("kind", String(8), nullable=False),
    Column("ordinal", Integer, nullable=False),
    Column("text_content", Text, nullable=False),
    Column("token_count", Integer, nullable=False, default=0, server_default="0"),
    Column("content_hash", LargeBinary(32), nullable=False),
    Column("created_at", DateTime(6), nullable=False, default=lambda: datetime.now(timezone.utc), server_default=func.now()),
    Column("updated_at", DateTime(6), nullable=False, default=lambda: datetime.now(timezone.utc), server_default=func.now()),
)

chunk_span_table = Table(
    "chunk_span",
    metadata,
    Column("tenant_id", String(36), primary_key=True),
    Column("id", String(36), primary_key=True),
    Column("build_id", String(36), nullable=False),
    Column("chunk_id", String(36), nullable=False),
    Column("unit_id", String(36), nullable=False),
    Column("ordinal", Integer, nullable=False),
    Column("start_cp", Integer, nullable=False),
    Column("end_cp", Integer, nullable=False),
    Column("location", JSON, nullable=True),
    Column("created_at", DateTime(6), nullable=False, default=lambda: datetime.now(timezone.utc), server_default=func.now()),
    Column("updated_at", DateTime(6), nullable=False, default=lambda: datetime.now(timezone.utc), server_default=func.now()),
)

representation_table = Table(
    "representation",
    metadata,
    Column("tenant_id", String(36), primary_key=True),
    Column("id", String(36), primary_key=True),
    Column("build_id", String(36), nullable=False),
    Column("chunk_id", String(36), nullable=False),
    Column("kind", String(16), nullable=False),
    Column("ordinal", Integer, nullable=False, default=0, server_default="0"),
    Column("text_content", Text, nullable=False),
    Column("text_hash", LargeBinary(32), nullable=False),
    Column("model_fingerprint", String(255), nullable=True),
    Column("dims", Integer, nullable=True),
    Column("vector_bytes", LargeBinary, nullable=True),
    Column("vector_hash", LargeBinary(32), nullable=True),
    Column("created_at", DateTime(6), nullable=False, default=lambda: datetime.now(timezone.utc), server_default=func.now()),
    Column("updated_at", DateTime(6), nullable=False, default=lambda: datetime.now(timezone.utc), server_default=func.now()),
)

job_table = Table(
    "job",
    metadata,
    Column("tenant_id", String(36), primary_key=True),
    Column("id", String(36), primary_key=True),
    Column("kind", String(24), nullable=False),
    Column("target_id", String(36), nullable=False),
    Column("requested_by", String(36), nullable=False),
    Column("idempotency_key", String(128), nullable=False),
    Column("request_hash", LargeBinary(32), nullable=False),
    Column("status", String(24), nullable=False, default="QUEUED", server_default="QUEUED"),
    Column("stage", String(32), nullable=False, default="PARSING", server_default="PARSING"),
    Column("attempt", Integer, nullable=False, default=0, server_default="0"),
    Column("available_at", DateTime(6), nullable=False, default=lambda: datetime.now(timezone.utc), server_default=func.now()),
    Column("lease_until", DateTime(6), nullable=True),
    Column("lease_epoch", BigInteger, nullable=False, default=0, server_default="0"),
    Column("fence", BigInteger, nullable=True),
    Column("checkpoint", JSON, nullable=True),
    Column("error_code", String(64), nullable=True),
    Column("cancel_requested_at", DateTime(6), nullable=True),
    Column("created_at", DateTime(6), nullable=False, default=lambda: datetime.now(timezone.utc), server_default=func.now()),
    Column("updated_at", DateTime(6), nullable=False, default=lambda: datetime.now(timezone.utc), server_default=func.now()),
)

conversation_table = Table(
    "conversation",
    metadata,
    Column("tenant_id", String(36), primary_key=True),
    Column("id", String(36), primary_key=True),
    Column("owner_id", String(36), nullable=False),
    Column("title", String(255), nullable=True),
    Column("status", String(16), nullable=False, default="ACTIVE", server_default="ACTIVE"),
    Column("created_at", DateTime(6), nullable=False, default=lambda: datetime.now(timezone.utc), server_default=func.now()),
    Column("updated_at", DateTime(6), nullable=False, default=lambda: datetime.now(timezone.utc), server_default=func.now()),
)

context_segment_table = Table(
    "context_segment",
    metadata,
    Column("tenant_id", String(36), primary_key=True),
    Column("id", String(36), primary_key=True),
    Column("conversation_id", String(36), nullable=False),
    Column("segment_no", Integer, nullable=False, default=1, server_default="1"),
    Column("scope_mode", String(8), nullable=False, default="ALL", server_default="ALL"),
    # Persistent, per-segment context for new Agent turns; the version advances
    # only on initialization and compaction (turn_no tracks normal append).
    Column("context_payload", JSON, nullable=True),
    Column("context_version", Integer, nullable=False, default=0, server_default="0"),
    Column("context_through_turn", Integer, nullable=False, default=0, server_default="0"),
    Column("context_boundary_turn", Integer, nullable=False, default=0, server_default="0"),
    Column("context_last_run_id", String(36), nullable=True),
    Column("created_at", DateTime(6), nullable=False, default=lambda: datetime.now(timezone.utc), server_default=func.now()),
    Column("updated_at", DateTime(6), nullable=False, default=lambda: datetime.now(timezone.utc), server_default=func.now()),
)

segment_kb_table = Table(
    "segment_kb",
    metadata,
    Column("tenant_id", String(36), primary_key=True),
    Column("id", String(36), primary_key=True),
    Column("segment_id", String(36), nullable=False),
    Column("kb_id", String(36), nullable=False),
    Column("created_at", DateTime(6), nullable=False, default=lambda: datetime.now(timezone.utc), server_default=func.now()),
    Column("updated_at", DateTime(6), nullable=False, default=lambda: datetime.now(timezone.utc), server_default=func.now()),
)

qa_run_table = Table(
    "qa_run",
    metadata,
    Column("tenant_id", String(36), primary_key=True),
    Column("id", String(36), primary_key=True),
    Column("owner_id", String(36), nullable=False),
    Column("conversation_id", String(36), nullable=False),
    Column("segment_id", String(36), nullable=False),
    Column("turn_no", Integer, nullable=False, default=1, server_default="1"),
    Column("config_id", String(36), nullable=True),
    Column("mode", String(8), nullable=False, default="RAG", server_default="RAG"),
    Column("strict_mode", Boolean, nullable=False, default=False, server_default="0"),
    Column("status", String(32), nullable=False, default="QUEUED", server_default="QUEUED"),
    Column("stage", String(32), nullable=False, default="ROUTING", server_default="ROUTING"),
    Column("idempotency_key", String(128), nullable=False),
    Column("request_hash", LargeBinary(32), nullable=False),
    Column("question", Text, nullable=True),
    Column("result_json", JSON, nullable=True),
    Column("security_epoch", BigInteger, nullable=False, default=0, server_default="0"),
    Column("budget_json", JSON, nullable=True),
    Column("terminal_reason", String(64), nullable=True),
    Column("cancel_requested_at", DateTime(6), nullable=True),
    Column("started_at", DateTime(6), nullable=True),
    Column("finished_at", DateTime(6), nullable=True),
    Column("created_at", DateTime(6), nullable=False, default=lambda: datetime.now(timezone.utc), server_default=func.now()),
    Column("updated_at", DateTime(6), nullable=False, default=lambda: datetime.now(timezone.utc), server_default=func.now()),
)

run_source_table = Table(
    "run_source",
    metadata,
    Column("tenant_id", String(36), primary_key=True),
    Column("id", String(36), primary_key=True),
    Column("run_id", String(36), nullable=False),
    # DOCUMENT 保持原文档外键；TOOL 使用独立快照，不创建假 document/build。
    Column("document_id", String(36), nullable=True),
    Column("build_id", String(36), nullable=True),
    Column("unit_id", String(36), nullable=True),
    Column("evidence_id", String(32), nullable=True),
    Column("source_type", String(16), nullable=False, default="DOCUMENT", server_default="DOCUMENT"),
    Column("tool_name", String(80), nullable=True),
    Column("source_snapshot", JSON, nullable=True),
    Column("purpose", String(16), nullable=False, default="RETRIEVED", server_default="RETRIEVED"),
    Column("span_json", JSON, nullable=True),
    Column("payload_hash", LargeBinary(32), nullable=True),
    Column("send_status", String(16), nullable=False, default="PREPARED", server_default="PREPARED"),
    Column("created_at", DateTime(6), nullable=False, default=lambda: datetime.now(timezone.utc), server_default=func.now()),
    Column("updated_at", DateTime(6), nullable=False, default=lambda: datetime.now(timezone.utc), server_default=func.now()),
)

run_dependency_table = Table(
    "run_dependency",
    metadata,
    Column("tenant_id", String(36), primary_key=True),
    Column("id", String(36), primary_key=True),
    Column("run_id", String(36), nullable=False),
    Column("prior_run_id", String(36), nullable=False),
    Column("created_at", DateTime(6), nullable=False, default=lambda: datetime.now(timezone.utc), server_default=func.now()),
    Column("updated_at", DateTime(6), nullable=False, default=lambda: datetime.now(timezone.utc), server_default=func.now()),
)

run_event_table = Table(
    "run_event",
    metadata,
    Column("tenant_id", String(36), primary_key=True),
    Column("id", String(36), primary_key=True),
    Column("run_id", String(36), nullable=False),
    Column("sequence_no", Integer, nullable=False, default=1, server_default="1"),
    Column("stage", String(32), nullable=False),
    Column("duration_ms", BigInteger, nullable=True),
    Column("payload", JSON, nullable=True),
    Column("error_code", String(64), nullable=True),
    Column("created_at", DateTime(6), nullable=False, default=lambda: datetime.now(timezone.utc), server_default=func.now()),
    Column("updated_at", DateTime(6), nullable=False, default=lambda: datetime.now(timezone.utc), server_default=func.now()),
)

tenant_budget_table = Table(
    "tenant_budget",
    metadata,
    Column("tenant_id", String(36), primary_key=True),
    Column("id", String(36), primary_key=True),
    Column("period_month", String(7), nullable=False),
    Column("currency", String(3), nullable=False, default="CNY", server_default="CNY"),
    Column("limit_amount", Numeric(18, 6), nullable=False),
    Column("reserved_amount", Numeric(18, 6), nullable=False, default=0, server_default="0"),
    Column("settled_amount", Numeric(18, 6), nullable=False, default=0, server_default="0"),
    Column("created_at", DateTime(6), nullable=False, default=lambda: datetime.now(timezone.utc), server_default=func.now()),
    Column("updated_at", DateTime(6), nullable=False, default=lambda: datetime.now(timezone.utc), server_default=func.now()),
)

budget_reservation_table = Table(
    "budget_reservation",
    metadata,
    Column("tenant_id", String(36), primary_key=True),
    Column("id", String(36), primary_key=True),
    Column("budget_id", String(36), nullable=False),
    Column("caller_environment", String(128), nullable=False),
    Column("caller_call_id", String(36), nullable=False),
    Column("status", String(16), nullable=False),
    Column("reserved_amount", Numeric(18, 6), nullable=False),
    Column("settled_amount", Numeric(18, 6), nullable=True),
    Column("expires_at", DateTime(6), nullable=False),
    Column("created_at", DateTime(6), nullable=False, default=lambda: datetime.now(timezone.utc), server_default=func.now()),
    Column("updated_at", DateTime(6), nullable=False, default=lambda: datetime.now(timezone.utc), server_default=func.now()),
)

model_call_table = Table(
    "model_call",
    metadata,
    Column("tenant_id", String(36), primary_key=True),
    Column("id", String(36), primary_key=True),
    Column("run_id", String(36), nullable=True),
    Column("job_id", String(36), nullable=True),
    Column("purpose", String(24), nullable=False),
    Column("model_config_id", String(36), nullable=False),
    Column("attempt_no", Integer, nullable=False, default=1, server_default="1"),
    Column("status", String(16), nullable=False, default="SUCCESS", server_default="SUCCESS"),
    Column("reserved_amount", Numeric(18, 6), nullable=False, default=0, server_default="0"),
    Column("settled_amount", Numeric(18, 6), nullable=True),
    Column("input_tokens", BigInteger, nullable=True),
    Column("output_tokens", BigInteger, nullable=True),
    Column("provider_request_id", String(255), nullable=True),
    Column("pricing_snapshot", JSON, nullable=False),
    Column("usage_detail", JSON, nullable=True),
    Column("reservation_authority", String(128), nullable=False),
    Column("reservation_id", String(36), nullable=True),
    Column("created_at", DateTime(6), nullable=False, default=lambda: datetime.now(timezone.utc), server_default=func.now()),
    Column("updated_at", DateTime(6), nullable=False, default=lambda: datetime.now(timezone.utc), server_default=func.now()),
    CheckConstraint("((run_id IS NULL) <> (job_id IS NULL))", name="ck_model_call_owner"),
)

feedback_table = Table(
    "feedback",
    metadata,
    Column("tenant_id", String(36), primary_key=True),
    Column("id", String(36), primary_key=True),
    Column("run_id", String(36), nullable=False),
    Column("submitted_by", String(36), nullable=False),
    Column("rating", String(8), nullable=True),
    Column("issue_type", String(32), nullable=True),
    Column("comment_text", Text, nullable=True),
    Column("status", String(16), nullable=False, default="OPEN", server_default="OPEN"),
    Column("resolution_text", Text, nullable=True),
    Column("created_at", DateTime(6), nullable=False, default=lambda: datetime.now(timezone.utc), server_default=func.now()),
    Column("updated_at", DateTime(6), nullable=False, default=lambda: datetime.now(timezone.utc), server_default=func.now()),
)

audit_event_table = Table(
    "audit_event",
    metadata,
    Column("tenant_id", String(36), primary_key=True),
    Column("id", String(36), primary_key=True),
    Column("actor_id", String(36), nullable=True),
    Column("action", String(64), nullable=False),
    Column("object_type", String(32), nullable=False),
    Column("object_id", String(36), nullable=False),
    Column("outcome", String(16), nullable=False, default="SUCCESS", server_default="SUCCESS"),
    Column("metadata_json", JSON, nullable=True),
    Column("created_at", DateTime(6), nullable=False, default=lambda: datetime.now(timezone.utc), server_default=func.now()),
    Column("updated_at", DateTime(6), nullable=False, default=lambda: datetime.now(timezone.utc), server_default=func.now()),
)

deletion_record_table = Table(
    "deletion_record",
    metadata,
    Column("tenant_id", String(36), primary_key=True),
    Column("id", String(36), primary_key=True),
    Column("document_id", String(36), nullable=False),
    Column("origin_system", String(128), nullable=False, default="local", server_default="local"),
    Column("requested_by", String(36), nullable=False),
    Column("status", String(24), nullable=False, default="PENDING", server_default="PENDING"),
    Column("domains_json", JSON, nullable=False),
    Column("backup_expiry_at", DateTime(6), nullable=False),
    Column("completed_at", DateTime(6), nullable=True),
    Column("created_at", DateTime(6), nullable=False, default=lambda: datetime.now(timezone.utc), server_default=func.now()),
    Column("updated_at", DateTime(6), nullable=False, default=lambda: datetime.now(timezone.utc), server_default=func.now()),
)

eval_dataset_version_table = Table(
    "eval_dataset_version",
    metadata,
    Column("tenant_id", String(36), primary_key=True),
    Column("id", String(36), primary_key=True),
    Column("name", String(128), nullable=False),
    Column("revision", Integer, nullable=False),
    Column("status", String(16), nullable=False, default="DRAFT", server_default="DRAFT"),
    Column("manifest_hash", LargeBinary(32), nullable=True),
    Column("frozen_at", DateTime(6), nullable=True),
    Column("created_at", DateTime(6), nullable=False, default=lambda: datetime.now(timezone.utc), server_default=func.now()),
    Column("updated_at", DateTime(6), nullable=False, default=lambda: datetime.now(timezone.utc), server_default=func.now()),
)

eval_case_table = Table(
    "eval_case",
    metadata,
    Column("tenant_id", String(36), primary_key=True),
    Column("id", String(36), primary_key=True),
    Column("dataset_version_id", String(36), nullable=False),
    Column("case_key", String(128), nullable=False),
    Column("split", String(8), nullable=False),
    Column("group_key", String(128), nullable=False),
    Column("category", String(32), nullable=False),
    Column("question_json", JSON, nullable=False),
    Column("persona_json", JSON, nullable=False),
    Column("expected_json", JSON, nullable=False),
    Column("annotation_status", String(16), nullable=False),
    Column("created_at", DateTime(6), nullable=False, default=lambda: datetime.now(timezone.utc), server_default=func.now()),
    Column("updated_at", DateTime(6), nullable=False, default=lambda: datetime.now(timezone.utc), server_default=func.now()),
)

eval_gold_table = Table(
    "eval_gold",
    metadata,
    Column("tenant_id", String(36), primary_key=True),
    Column("id", String(36), primary_key=True),
    Column("case_id", String(36), nullable=False),
    Column("fact_group", String(128), nullable=False),
    Column("alternative_no", Integer, nullable=False),
    Column("origin_system", String(128), nullable=False),
    Column("origin_tenant_id", String(36), nullable=False),
    Column("origin_document_id", String(36), nullable=False),
    Column("version_hash", LargeBinary(32), nullable=False),
    Column("anchor_json", JSON, nullable=False),
    Column("relevance_grade", Integer, nullable=False),
    Column("created_at", DateTime(6), nullable=False, default=lambda: datetime.now(timezone.utc), server_default=func.now()),
    Column("updated_at", DateTime(6), nullable=False, default=lambda: datetime.now(timezone.utc), server_default=func.now()),
)

eval_snapshot_table = Table(
    "eval_snapshot",
    metadata,
    Column("tenant_id", String(36), primary_key=True),
    Column("id", String(36), primary_key=True),
    Column("name", String(128), nullable=False),
    Column("manifest_hash", LargeBinary(32), nullable=False),
    Column("manifest_json", JSON, nullable=False),
    Column("status", String(16), nullable=False),
    Column("environment_id", String(128), nullable=False),
    Column("created_at", DateTime(6), nullable=False, default=lambda: datetime.now(timezone.utc), server_default=func.now()),
    Column("updated_at", DateTime(6), nullable=False, default=lambda: datetime.now(timezone.utc), server_default=func.now()),
)

eval_snapshot_document_table = Table(
    "eval_snapshot_document",
    metadata,
    Column("tenant_id", String(36), primary_key=True),
    Column("id", String(36), primary_key=True),
    Column("snapshot_id", String(36), nullable=False),
    Column("document_id", String(36), nullable=False),
    Column("version_id", String(36), nullable=False),
    Column("origin_system", String(128), nullable=False),
    Column("origin_tenant_id", String(36), nullable=False),
    Column("origin_document_id", String(36), nullable=False),
    Column("origin_version_hash", LargeBinary(32), nullable=False),
    Column("created_at", DateTime(6), nullable=False, default=lambda: datetime.now(timezone.utc), server_default=func.now()),
    Column("updated_at", DateTime(6), nullable=False, default=lambda: datetime.now(timezone.utc), server_default=func.now()),
)

eval_run_table = Table(
    "eval_run",
    metadata,
    Column("tenant_id", String(36), primary_key=True),
    Column("id", String(36), primary_key=True),
    Column("dataset_version_id", String(36), nullable=False),
    Column("snapshot_id", String(36), nullable=False),
    Column("config_id", String(36), nullable=False),
    Column("job_id", String(36), nullable=False),
    Column("status", String(24), nullable=False),
    Column("manifest_json", JSON, nullable=False),
    Column("metrics_json", JSON, nullable=True),
    Column("created_at", DateTime(6), nullable=False, default=lambda: datetime.now(timezone.utc), server_default=func.now()),
    Column("updated_at", DateTime(6), nullable=False, default=lambda: datetime.now(timezone.utc), server_default=func.now()),
)

eval_result_table = Table(
    "eval_result",
    metadata,
    Column("tenant_id", String(36), primary_key=True),
    Column("id", String(36), primary_key=True),
    Column("eval_run_id", String(36), nullable=False),
    Column("dataset_version_id", String(36), nullable=False),
    Column("case_id", String(36), nullable=False),
    Column("repeat_no", Integer, nullable=False, default=0, server_default="0"),
    Column("qa_run_id", String(36), nullable=True),
    Column("status", String(24), nullable=False),
    Column("metrics_json", JSON, nullable=True),
    Column("judge_json", JSON, nullable=True),
    Column("error_code", String(64), nullable=True),
    Column("created_at", DateTime(6), nullable=False, default=lambda: datetime.now(timezone.utc), server_default=func.now()),
    Column("updated_at", DateTime(6), nullable=False, default=lambda: datetime.now(timezone.utc), server_default=func.now()),
)

eval_review_table = Table(
    "eval_review",
    metadata,
    Column("tenant_id", String(36), primary_key=True),
    Column("id", String(36), primary_key=True),
    Column("result_id", String(36), nullable=False),
    Column("reviewer_id", String(36), nullable=False),
    Column("labels_json", JSON, nullable=False),
    Column("reason_text", Text, nullable=True),
    Column("created_at", DateTime(6), nullable=False, default=lambda: datetime.now(timezone.utc), server_default=func.now()),
    Column("updated_at", DateTime(6), nullable=False, default=lambda: datetime.now(timezone.utc), server_default=func.now()),
)


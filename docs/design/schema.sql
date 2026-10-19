-- 设计版 DDL v1.0，目标 MySQL 8.4；该文件用于描述目标结构，不代表已经执行，也不是可直接回放的迁移记录。
-- 文件中不包含建库/删库命令、密钥、seed 账号密码或公司业务内容，避免设计文档携带环境敏感信息。
-- 所有时间字段使用 UTC DATETIME(6)；应用连接必须设置 +00:00 时区，避免跨环境写入本地时区造成排序/过期判断偏差。
-- job.target_id 与 audit 对象 ID 属于多态引用，无法用单一外键表达，因此由服务层按 kind/object_type 强制校验。

CREATE TABLE tenant (
  id CHAR(36) CHARACTER SET ascii COLLATE ascii_bin NOT NULL,
  code VARCHAR(64) CHARACTER SET ascii COLLATE ascii_bin NOT NULL,
  name VARCHAR(128) NOT NULL,
  security_epoch BIGINT NOT NULL DEFAULT 0,
  active_retrieval_config_id CHAR(36) CHARACTER SET ascii COLLATE ascii_bin NULL,
  created_at DATETIME(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6),
  PRIMARY KEY (id),
  UNIQUE KEY uq_tenant_code (code)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;

CREATE TABLE app_user (
  tenant_id CHAR(36) CHARACTER SET ascii COLLATE ascii_bin NOT NULL,
  id CHAR(36) CHARACTER SET ascii COLLATE ascii_bin NOT NULL,
  login_name VARCHAR(128) NOT NULL,
  password_hash VARCHAR(255) NOT NULL,
  is_admin BOOLEAN NOT NULL DEFAULT FALSE,
  status VARCHAR(16) NOT NULL DEFAULT 'ACTIVE',
  created_at DATETIME(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6),
  updated_at DATETIME(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6) ON UPDATE CURRENT_TIMESTAMP(6),
  PRIMARY KEY (tenant_id, id),
  UNIQUE KEY uq_user_login (tenant_id, login_name),
  CHECK (status IN ('ACTIVE','DISABLED')),
  CONSTRAINT fk_1 FOREIGN KEY (tenant_id) REFERENCES tenant (id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;

CREATE TABLE auth_session (
  tenant_id CHAR(36) CHARACTER SET ascii COLLATE ascii_bin NOT NULL,
  id CHAR(36) CHARACTER SET ascii COLLATE ascii_bin NOT NULL,
  user_id CHAR(36) CHARACTER SET ascii COLLATE ascii_bin NOT NULL,
  token_hash BINARY(32) NOT NULL,
  expires_at DATETIME(6) NOT NULL,
  revoked_at DATETIME(6) NULL,
  created_at DATETIME(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6),
  updated_at DATETIME(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6) ON UPDATE CURRENT_TIMESTAMP(6),
  PRIMARY KEY (tenant_id, id),
  UNIQUE KEY uq_session_token (token_hash),
  KEY ix_session_user (tenant_id,user_id,expires_at),
  CONSTRAINT fk_2 FOREIGN KEY (tenant_id) REFERENCES tenant (id),
  CONSTRAINT fk_3 FOREIGN KEY (tenant_id,user_id) REFERENCES app_user (tenant_id,id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;

CREATE TABLE user_group (
  tenant_id CHAR(36) CHARACTER SET ascii COLLATE ascii_bin NOT NULL,
  id CHAR(36) CHARACTER SET ascii COLLATE ascii_bin NOT NULL,
  name VARCHAR(128) NOT NULL,
  created_at DATETIME(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6),
  updated_at DATETIME(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6) ON UPDATE CURRENT_TIMESTAMP(6),
  PRIMARY KEY (tenant_id, id),
  UNIQUE KEY uq_group_name (tenant_id,name),
  CONSTRAINT fk_4 FOREIGN KEY (tenant_id) REFERENCES tenant (id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;

CREATE TABLE group_member (
  tenant_id CHAR(36) CHARACTER SET ascii COLLATE ascii_bin NOT NULL,
  id CHAR(36) CHARACTER SET ascii COLLATE ascii_bin NOT NULL,
  group_id CHAR(36) CHARACTER SET ascii COLLATE ascii_bin NOT NULL,
  user_id CHAR(36) CHARACTER SET ascii COLLATE ascii_bin NOT NULL,
  created_at DATETIME(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6),
  updated_at DATETIME(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6) ON UPDATE CURRENT_TIMESTAMP(6),
  PRIMARY KEY (tenant_id, id),
  UNIQUE KEY uq_member (tenant_id,group_id,user_id),
  KEY ix_member_user (tenant_id,user_id),
  CONSTRAINT fk_5 FOREIGN KEY (tenant_id) REFERENCES tenant (id),
  CONSTRAINT fk_6 FOREIGN KEY (tenant_id,group_id) REFERENCES user_group (tenant_id,id),
  CONSTRAINT fk_7 FOREIGN KEY (tenant_id,user_id) REFERENCES app_user (tenant_id,id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;

CREATE TABLE knowledge_base (
  tenant_id CHAR(36) CHARACTER SET ascii COLLATE ascii_bin NOT NULL,
  id CHAR(36) CHARACTER SET ascii COLLATE ascii_bin NOT NULL,
  name VARCHAR(255) NOT NULL,
  status VARCHAR(16) NOT NULL DEFAULT 'LIVE',
  row_version BIGINT NOT NULL DEFAULT 0,
  created_at DATETIME(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6),
  updated_at DATETIME(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6) ON UPDATE CURRENT_TIMESTAMP(6),
  PRIMARY KEY (tenant_id, id),
  CHECK (status IN ('LIVE','DISABLED')),
  CONSTRAINT fk_8 FOREIGN KEY (tenant_id) REFERENCES tenant (id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;

CREATE TABLE kb_grant (
  tenant_id CHAR(36) CHARACTER SET ascii COLLATE ascii_bin NOT NULL,
  id CHAR(36) CHARACTER SET ascii COLLATE ascii_bin NOT NULL,
  kb_id CHAR(36) CHARACTER SET ascii COLLATE ascii_bin NOT NULL,
  principal_type VARCHAR(8) NOT NULL,
  user_id CHAR(36) CHARACTER SET ascii COLLATE ascii_bin NULL,
  group_id CHAR(36) CHARACTER SET ascii COLLATE ascii_bin NULL,
  principal_key VARCHAR(80) CHARACTER SET ascii COLLATE ascii_bin GENERATED ALWAYS AS (CONCAT(principal_type,':',COALESCE(user_id,group_id,'*'))) STORED,
  created_at DATETIME(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6),
  updated_at DATETIME(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6) ON UPDATE CURRENT_TIMESTAMP(6),
  PRIMARY KEY (tenant_id, id),
  UNIQUE KEY uq_kb_grant (tenant_id,kb_id,principal_key),
  CHECK ((principal_type='USER' AND user_id IS NOT NULL AND group_id IS NULL) OR (principal_type='GROUP' AND group_id IS NOT NULL AND user_id IS NULL) OR (principal_type='ALL' AND user_id IS NULL AND group_id IS NULL)),
  CONSTRAINT fk_9 FOREIGN KEY (tenant_id) REFERENCES tenant (id),
  CONSTRAINT fk_10 FOREIGN KEY (tenant_id,kb_id) REFERENCES knowledge_base (tenant_id,id),
  CONSTRAINT fk_11 FOREIGN KEY (tenant_id,user_id) REFERENCES app_user (tenant_id,id),
  CONSTRAINT fk_12 FOREIGN KEY (tenant_id,group_id) REFERENCES user_group (tenant_id,id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;

CREATE TABLE document (
  tenant_id CHAR(36) CHARACTER SET ascii COLLATE ascii_bin NOT NULL,
  id CHAR(36) CHARACTER SET ascii COLLATE ascii_bin NOT NULL,
  kb_id CHAR(36) CHARACTER SET ascii COLLATE ascii_bin NOT NULL,
  title VARCHAR(512) NULL,
  acl_mode VARCHAR(16) NOT NULL DEFAULT 'RESTRICT',
  status VARCHAR(20) NOT NULL DEFAULT 'LIVE',
  active_build_id CHAR(36) CHARACTER SET ascii COLLATE ascii_bin NULL,
  lifecycle_epoch BIGINT NOT NULL DEFAULT 0,
  row_version BIGINT NOT NULL DEFAULT 0,
  deleted_at DATETIME(6) NULL,
  created_at DATETIME(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6),
  updated_at DATETIME(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6) ON UPDATE CURRENT_TIMESTAMP(6),
  PRIMARY KEY (tenant_id, id),
  KEY ix_doc_kb (tenant_id,kb_id,status),
  CHECK (acl_mode IN ('INHERIT','RESTRICT')),
  CHECK (status IN ('LIVE','TRASHED','PURGE_PENDING','PURGED')),
  CONSTRAINT fk_13 FOREIGN KEY (tenant_id) REFERENCES tenant (id),
  CONSTRAINT fk_14 FOREIGN KEY (tenant_id,kb_id) REFERENCES knowledge_base (tenant_id,id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;

CREATE TABLE document_grant (
  tenant_id CHAR(36) CHARACTER SET ascii COLLATE ascii_bin NOT NULL,
  id CHAR(36) CHARACTER SET ascii COLLATE ascii_bin NOT NULL,
  document_id CHAR(36) CHARACTER SET ascii COLLATE ascii_bin NOT NULL,
  principal_type VARCHAR(8) NOT NULL,
  user_id CHAR(36) CHARACTER SET ascii COLLATE ascii_bin NULL,
  group_id CHAR(36) CHARACTER SET ascii COLLATE ascii_bin NULL,
  principal_key VARCHAR(80) CHARACTER SET ascii COLLATE ascii_bin GENERATED ALWAYS AS (CONCAT(principal_type,':',COALESCE(user_id,group_id,'*'))) STORED,
  created_at DATETIME(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6),
  updated_at DATETIME(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6) ON UPDATE CURRENT_TIMESTAMP(6),
  PRIMARY KEY (tenant_id, id),
  UNIQUE KEY uq_document_grant (tenant_id,document_id,principal_key),
  CHECK ((principal_type='USER' AND user_id IS NOT NULL AND group_id IS NULL) OR (principal_type='GROUP' AND group_id IS NOT NULL AND user_id IS NULL)),
  CONSTRAINT fk_15 FOREIGN KEY (tenant_id) REFERENCES tenant (id),
  CONSTRAINT fk_16 FOREIGN KEY (tenant_id,document_id) REFERENCES document (tenant_id,id),
  CONSTRAINT fk_17 FOREIGN KEY (tenant_id,user_id) REFERENCES app_user (tenant_id,id),
  CONSTRAINT fk_18 FOREIGN KEY (tenant_id,group_id) REFERENCES user_group (tenant_id,id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;

CREATE TABLE artifact (
  tenant_id CHAR(36) CHARACTER SET ascii COLLATE ascii_bin NOT NULL,
  id CHAR(36) CHARACTER SET ascii COLLATE ascii_bin NOT NULL,
  document_id CHAR(36) CHARACTER SET ascii COLLATE ascii_bin NOT NULL,
  kind VARCHAR(32) NOT NULL,
  storage_key VARCHAR(512) CHARACTER SET ascii COLLATE ascii_bin NOT NULL,
  sha256 BINARY(32) NOT NULL,
  size_bytes BIGINT NOT NULL,
  status VARCHAR(16) NOT NULL DEFAULT 'READY',
  created_at DATETIME(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6),
  updated_at DATETIME(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6) ON UPDATE CURRENT_TIMESTAMP(6),
  PRIMARY KEY (tenant_id, id),
  UNIQUE KEY uq_artifact_key (storage_key),
  UNIQUE KEY uq_artifact_doc (tenant_id,document_id,id),
  CHECK (size_bytes >= 0),
  CONSTRAINT fk_19 FOREIGN KEY (tenant_id) REFERENCES tenant (id),
  CONSTRAINT fk_20 FOREIGN KEY (tenant_id,document_id) REFERENCES document (tenant_id,id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;

CREATE TABLE document_version (
  tenant_id CHAR(36) CHARACTER SET ascii COLLATE ascii_bin NOT NULL,
  id CHAR(36) CHARACTER SET ascii COLLATE ascii_bin NOT NULL,
  document_id CHAR(36) CHARACTER SET ascii COLLATE ascii_bin NOT NULL,
  version_no INT NOT NULL,
  source_artifact_id CHAR(36) CHARACTER SET ascii COLLATE ascii_bin NULL,
  source_hash BINARY(32) NOT NULL,
  original_name VARCHAR(512) NULL,
  media_type VARCHAR(128) NOT NULL,
  created_by CHAR(36) CHARACTER SET ascii COLLATE ascii_bin NOT NULL,
  created_at DATETIME(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6),
  updated_at DATETIME(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6) ON UPDATE CURRENT_TIMESTAMP(6),
  PRIMARY KEY (tenant_id, id),
  UNIQUE KEY uq_doc_version (tenant_id,document_id,version_no),
  UNIQUE KEY uq_version_doc (tenant_id,document_id,id),
  CHECK (version_no > 0),
  CONSTRAINT fk_21 FOREIGN KEY (tenant_id) REFERENCES tenant (id),
  CONSTRAINT fk_22 FOREIGN KEY (tenant_id,document_id) REFERENCES document (tenant_id,id),
  CONSTRAINT fk_23 FOREIGN KEY (tenant_id,document_id,source_artifact_id) REFERENCES artifact (tenant_id,document_id,id),
  CONSTRAINT fk_24 FOREIGN KEY (tenant_id,created_by) REFERENCES app_user (tenant_id,id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;

CREATE TABLE config_revision (
  tenant_id CHAR(36) CHARACTER SET ascii COLLATE ascii_bin NOT NULL,
  id CHAR(36) CHARACTER SET ascii COLLATE ascii_bin NOT NULL,
  kind VARCHAR(24) NOT NULL,
  name VARCHAR(128) NOT NULL,
  revision INT NOT NULL,
  sha256 BINARY(32) NOT NULL,
  payload JSON NOT NULL,
  status VARCHAR(16) NOT NULL DEFAULT 'DRAFT',
  created_at DATETIME(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6),
  updated_at DATETIME(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6) ON UPDATE CURRENT_TIMESTAMP(6),
  PRIMARY KEY (tenant_id, id),
  UNIQUE KEY uq_config_revision (tenant_id,kind,name,revision),
  CHECK (revision > 0),
  CHECK (kind IN ('BUILD','RETRIEVAL','MODEL','PROMPT','LIMITS','EVAL')),
  CONSTRAINT fk_25 FOREIGN KEY (tenant_id) REFERENCES tenant (id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;

CREATE TABLE index_build (
  tenant_id CHAR(36) CHARACTER SET ascii COLLATE ascii_bin NOT NULL,
  id CHAR(36) CHARACTER SET ascii COLLATE ascii_bin NOT NULL,
  document_id CHAR(36) CHARACTER SET ascii COLLATE ascii_bin NOT NULL,
  version_id CHAR(36) CHARACTER SET ascii COLLATE ascii_bin NOT NULL,
  build_no INT NOT NULL,
  config_id CHAR(36) CHARACTER SET ascii COLLATE ascii_bin NOT NULL,
  status VARCHAR(24) NOT NULL DEFAULT 'QUEUED',
  model_fingerprint VARCHAR(255) NULL,
  embedding_dims INT NULL,
  capabilities JSON NOT NULL,
  quality_warnings JSON NULL,
  review_reason VARCHAR(2000) NULL,
  manifest_hash BINARY(32) NULL,
  fence BIGINT NOT NULL,
  published_at DATETIME(6) NULL,
  created_at DATETIME(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6),
  updated_at DATETIME(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6) ON UPDATE CURRENT_TIMESTAMP(6),
  PRIMARY KEY (tenant_id, id),
  UNIQUE KEY uq_build_no (tenant_id,version_id,build_no),
  UNIQUE KEY uq_build_doc (tenant_id,document_id,id),
  CHECK (build_no > 0),
  CHECK (embedding_dims IS NULL OR embedding_dims > 0),
  CONSTRAINT fk_26 FOREIGN KEY (tenant_id) REFERENCES tenant (id),
  CONSTRAINT fk_27 FOREIGN KEY (tenant_id,document_id,version_id) REFERENCES document_version (tenant_id,document_id,id),
  CONSTRAINT fk_28 FOREIGN KEY (tenant_id,config_id) REFERENCES config_revision (tenant_id,id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;

CREATE TABLE source_unit (
  tenant_id CHAR(36) CHARACTER SET ascii COLLATE ascii_bin NOT NULL,
  id CHAR(36) CHARACTER SET ascii COLLATE ascii_bin NOT NULL,
  build_id CHAR(36) CHARACTER SET ascii COLLATE ascii_bin NOT NULL,
  unit_key VARCHAR(191) NOT NULL,
  kind VARCHAR(16) NOT NULL,
  text_content LONGTEXT NULL,
  location JSON NOT NULL,
  machine_generated BOOLEAN NOT NULL DEFAULT FALSE,
  artifact_id CHAR(36) CHARACTER SET ascii COLLATE ascii_bin NULL,
  content_hash BINARY(32) NOT NULL,
  created_at DATETIME(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6),
  updated_at DATETIME(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6) ON UPDATE CURRENT_TIMESTAMP(6),
  document_id CHAR(36) CHARACTER SET ascii COLLATE ascii_bin NOT NULL,
  PRIMARY KEY (tenant_id, id),
  UNIQUE KEY uq_unit_key (tenant_id,build_id,unit_key),
  UNIQUE KEY uq_unit_build (tenant_id,build_id,id),
  CHECK (kind IN ('TEXT','CODE','TABLE','IMAGE','HEADING')),
  CONSTRAINT fk_201 FOREIGN KEY (tenant_id) REFERENCES tenant (id),
  CONSTRAINT fk_202 FOREIGN KEY (tenant_id,document_id,build_id) REFERENCES index_build (tenant_id,document_id,id),
  CONSTRAINT fk_203 FOREIGN KEY (tenant_id,document_id,artifact_id) REFERENCES artifact (tenant_id,document_id,id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;

CREATE TABLE chunk (
  tenant_id CHAR(36) CHARACTER SET ascii COLLATE ascii_bin NOT NULL,
  id CHAR(36) CHARACTER SET ascii COLLATE ascii_bin NOT NULL,
  build_id CHAR(36) CHARACTER SET ascii COLLATE ascii_bin NOT NULL,
  parent_id CHAR(36) CHARACTER SET ascii COLLATE ascii_bin NULL,
  kind VARCHAR(8) NOT NULL,
  ordinal INT NOT NULL,
  text_content LONGTEXT NOT NULL,
  token_count INT NOT NULL,
  content_hash BINARY(32) NOT NULL,
  created_at DATETIME(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6),
  updated_at DATETIME(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6) ON UPDATE CURRENT_TIMESTAMP(6),
  PRIMARY KEY (tenant_id, id),
  UNIQUE KEY uq_chunk_build (tenant_id,build_id,id),
  UNIQUE KEY uq_chunk_ordinal (tenant_id,build_id,kind,ordinal),
  CHECK (kind IN ('PARENT','CHILD')),
  CHECK (token_count >= 0),
  CHECK (parent_id IS NULL OR parent_id <> id),
  CONSTRAINT fk_32 FOREIGN KEY (tenant_id) REFERENCES tenant (id),
  CONSTRAINT fk_33 FOREIGN KEY (tenant_id,build_id) REFERENCES index_build (tenant_id,id),
  CONSTRAINT fk_34 FOREIGN KEY (tenant_id,build_id,parent_id) REFERENCES chunk (tenant_id,build_id,id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;

CREATE TABLE chunk_span (
  tenant_id CHAR(36) CHARACTER SET ascii COLLATE ascii_bin NOT NULL,
  id CHAR(36) CHARACTER SET ascii COLLATE ascii_bin NOT NULL,
  build_id CHAR(36) CHARACTER SET ascii COLLATE ascii_bin NOT NULL,
  chunk_id CHAR(36) CHARACTER SET ascii COLLATE ascii_bin NOT NULL,
  unit_id CHAR(36) CHARACTER SET ascii COLLATE ascii_bin NOT NULL,
  ordinal INT NOT NULL,
  start_cp INT NOT NULL,
  end_cp INT NOT NULL,
  location JSON NULL,
  created_at DATETIME(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6),
  updated_at DATETIME(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6) ON UPDATE CURRENT_TIMESTAMP(6),
  PRIMARY KEY (tenant_id, id),
  UNIQUE KEY uq_span_order (tenant_id,chunk_id,ordinal),
  CHECK (start_cp >= 0 AND end_cp > start_cp),
  CONSTRAINT fk_35 FOREIGN KEY (tenant_id) REFERENCES tenant (id),
  CONSTRAINT fk_36 FOREIGN KEY (tenant_id,build_id,chunk_id) REFERENCES chunk (tenant_id,build_id,id),
  CONSTRAINT fk_37 FOREIGN KEY (tenant_id,build_id,unit_id) REFERENCES source_unit (tenant_id,build_id,id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;

CREATE TABLE representation (
  tenant_id CHAR(36) CHARACTER SET ascii COLLATE ascii_bin NOT NULL,
  id CHAR(36) CHARACTER SET ascii COLLATE ascii_bin NOT NULL,
  build_id CHAR(36) CHARACTER SET ascii COLLATE ascii_bin NOT NULL,
  chunk_id CHAR(36) CHARACTER SET ascii COLLATE ascii_bin NOT NULL,
  kind VARCHAR(16) NOT NULL,
  ordinal INT NOT NULL DEFAULT 0,
  text_content LONGTEXT NOT NULL,
  text_hash BINARY(32) NOT NULL,
  model_fingerprint VARCHAR(255) NULL,
  dims INT NULL,
  vector_bytes LONGBLOB NULL,
  vector_hash BINARY(32) NULL,
  created_at DATETIME(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6),
  updated_at DATETIME(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6) ON UPDATE CURRENT_TIMESTAMP(6),
  PRIMARY KEY (tenant_id, id),
  UNIQUE KEY uq_rep_kind (tenant_id,chunk_id,kind,ordinal),
  KEY ix_rep_build (tenant_id,build_id),
  CHECK (kind IN ('ORIGINAL','SUMMARY','QUESTION')),
  CHECK (ordinal >= 0 AND (kind='QUESTION' OR ordinal=0)),
  CHECK ((vector_bytes IS NULL AND dims IS NULL) OR (vector_bytes IS NOT NULL AND dims IS NOT NULL AND dims > 0 AND OCTET_LENGTH(vector_bytes)=4*dims)),
  CONSTRAINT fk_204 FOREIGN KEY (tenant_id) REFERENCES tenant (id),
  CONSTRAINT fk_205 FOREIGN KEY (tenant_id,build_id,chunk_id) REFERENCES chunk (tenant_id,build_id,id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;

CREATE TABLE job (
  tenant_id CHAR(36) CHARACTER SET ascii COLLATE ascii_bin NOT NULL,
  id CHAR(36) CHARACTER SET ascii COLLATE ascii_bin NOT NULL,
  kind VARCHAR(24) NOT NULL,
  target_id CHAR(36) CHARACTER SET ascii COLLATE ascii_bin NOT NULL,
  requested_by CHAR(36) CHARACTER SET ascii COLLATE ascii_bin NOT NULL,
  idempotency_key VARCHAR(128) CHARACTER SET ascii COLLATE ascii_bin NOT NULL,
  request_hash BINARY(32) NOT NULL,
  status VARCHAR(24) NOT NULL DEFAULT 'QUEUED',
  stage VARCHAR(32) NOT NULL,
  attempt INT NOT NULL DEFAULT 0,
  available_at DATETIME(6) NOT NULL,
  lease_until DATETIME(6) NULL,
  lease_epoch BIGINT NOT NULL DEFAULT 0,
  fence BIGINT NULL,
  checkpoint JSON NULL,
  error_code VARCHAR(64) NULL,
  cancel_requested_at DATETIME(6) NULL,
  created_at DATETIME(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6),
  updated_at DATETIME(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6) ON UPDATE CURRENT_TIMESTAMP(6),
  PRIMARY KEY (tenant_id, id),
  UNIQUE KEY uq_job_idempotency (tenant_id,kind,idempotency_key),
  KEY ix_job_claim (status,available_at,lease_until),
  CONSTRAINT chk_job_kind CHECK (kind IN ('INGEST','PURGE','RESTORE','EVALUATE','EXPORT','CLEANUP','QA_RUN','SEARCH')),
  CONSTRAINT fk_40 FOREIGN KEY (tenant_id) REFERENCES tenant (id),
  CONSTRAINT fk_41 FOREIGN KEY (tenant_id,requested_by) REFERENCES app_user (tenant_id,id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;

CREATE TABLE conversation (
  tenant_id CHAR(36) CHARACTER SET ascii COLLATE ascii_bin NOT NULL,
  id CHAR(36) CHARACTER SET ascii COLLATE ascii_bin NOT NULL,
  owner_id CHAR(36) CHARACTER SET ascii COLLATE ascii_bin NOT NULL,
  title VARCHAR(255) NULL,
  status VARCHAR(16) NOT NULL DEFAULT 'ACTIVE',
  created_at DATETIME(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6),
  updated_at DATETIME(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6) ON UPDATE CURRENT_TIMESTAMP(6),
  PRIMARY KEY (tenant_id, id),
  UNIQUE KEY uq_conversation_owner (tenant_id,owner_id,id),
  CONSTRAINT fk_42 FOREIGN KEY (tenant_id) REFERENCES tenant (id),
  CONSTRAINT fk_43 FOREIGN KEY (tenant_id,owner_id) REFERENCES app_user (tenant_id,id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;

CREATE TABLE context_segment (
  tenant_id CHAR(36) CHARACTER SET ascii COLLATE ascii_bin NOT NULL,
  id CHAR(36) CHARACTER SET ascii COLLATE ascii_bin NOT NULL,
  conversation_id CHAR(36) CHARACTER SET ascii COLLATE ascii_bin NOT NULL,
  segment_no INT NOT NULL,
  scope_mode VARCHAR(8) NOT NULL,
  context_payload JSON NULL,
  context_version INT NOT NULL DEFAULT 0,
  context_through_turn INT NOT NULL DEFAULT 0,
  context_boundary_turn INT NOT NULL DEFAULT 0,
  context_last_run_id CHAR(36) CHARACTER SET ascii COLLATE ascii_bin NULL,
  created_at DATETIME(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6),
  updated_at DATETIME(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6) ON UPDATE CURRENT_TIMESTAMP(6),
  PRIMARY KEY (tenant_id, id),
  UNIQUE KEY uq_segment_no (tenant_id,conversation_id,segment_no),
  UNIQUE KEY uq_segment_conv (tenant_id,conversation_id,id),
  CHECK (scope_mode IN ('ALL','SELECTED')),
  CONSTRAINT fk_44 FOREIGN KEY (tenant_id) REFERENCES tenant (id),
  CONSTRAINT fk_45 FOREIGN KEY (tenant_id,conversation_id) REFERENCES conversation (tenant_id,id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;

CREATE TABLE segment_kb (
  tenant_id CHAR(36) CHARACTER SET ascii COLLATE ascii_bin NOT NULL,
  id CHAR(36) CHARACTER SET ascii COLLATE ascii_bin NOT NULL,
  segment_id CHAR(36) CHARACTER SET ascii COLLATE ascii_bin NOT NULL,
  kb_id CHAR(36) CHARACTER SET ascii COLLATE ascii_bin NOT NULL,
  created_at DATETIME(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6),
  updated_at DATETIME(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6) ON UPDATE CURRENT_TIMESTAMP(6),
  PRIMARY KEY (tenant_id, id),
  UNIQUE KEY uq_segment_kb (tenant_id,segment_id,kb_id),
  CONSTRAINT fk_46 FOREIGN KEY (tenant_id) REFERENCES tenant (id),
  CONSTRAINT fk_47 FOREIGN KEY (tenant_id,segment_id) REFERENCES context_segment (tenant_id,id),
  CONSTRAINT fk_48 FOREIGN KEY (tenant_id,kb_id) REFERENCES knowledge_base (tenant_id,id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;

CREATE TABLE qa_run (
  tenant_id CHAR(36) CHARACTER SET ascii COLLATE ascii_bin NOT NULL,
  id CHAR(36) CHARACTER SET ascii COLLATE ascii_bin NOT NULL,
  owner_id CHAR(36) CHARACTER SET ascii COLLATE ascii_bin NOT NULL,
  conversation_id CHAR(36) CHARACTER SET ascii COLLATE ascii_bin NOT NULL,
  segment_id CHAR(36) CHARACTER SET ascii COLLATE ascii_bin NOT NULL,
  turn_no INT NOT NULL,
  config_id CHAR(36) CHARACTER SET ascii COLLATE ascii_bin NOT NULL,
  mode VARCHAR(8) NOT NULL,
  strict_mode BOOLEAN NOT NULL DEFAULT FALSE,
  status VARCHAR(32) NOT NULL DEFAULT 'QUEUED',
  stage VARCHAR(32) NOT NULL,
  idempotency_key VARCHAR(128) CHARACTER SET ascii COLLATE ascii_bin NOT NULL,
  request_hash BINARY(32) NOT NULL,
  question LONGTEXT NULL,
  result_json JSON NULL,
  security_epoch BIGINT NOT NULL,
  budget_json JSON NOT NULL,
  terminal_reason VARCHAR(64) NULL,
  cancel_requested_at DATETIME(6) NULL,
  started_at DATETIME(6) NULL,
  finished_at DATETIME(6) NULL,
  active_conversation_id CHAR(36) CHARACTER SET ascii COLLATE ascii_bin GENERATED ALWAYS AS (CASE WHEN status IN ('QUEUED','RUNNING') THEN conversation_id ELSE NULL END) STORED,
  created_at DATETIME(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6),
  updated_at DATETIME(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6) ON UPDATE CURRENT_TIMESTAMP(6),
  PRIMARY KEY (tenant_id, id),
  UNIQUE KEY uq_active_conversation (tenant_id,active_conversation_id),
  UNIQUE KEY uq_run_idempotency (tenant_id,owner_id,idempotency_key),
  UNIQUE KEY uq_run_turn (tenant_id,conversation_id,turn_no),
  CHECK (mode IN ('RAG','AGENT')),
  CONSTRAINT fk_49 FOREIGN KEY (tenant_id) REFERENCES tenant (id),
  CONSTRAINT fk_50 FOREIGN KEY (tenant_id,owner_id,conversation_id) REFERENCES conversation (tenant_id,owner_id,id),
  CONSTRAINT fk_51 FOREIGN KEY (tenant_id,conversation_id,segment_id) REFERENCES context_segment (tenant_id,conversation_id,id),
  CONSTRAINT fk_52 FOREIGN KEY (tenant_id,config_id) REFERENCES config_revision (tenant_id,id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;

CREATE TABLE run_source (
  tenant_id CHAR(36) CHARACTER SET ascii COLLATE ascii_bin NOT NULL,
  id CHAR(36) CHARACTER SET ascii COLLATE ascii_bin NOT NULL,
  run_id CHAR(36) CHARACTER SET ascii COLLATE ascii_bin NOT NULL,
  document_id CHAR(36) CHARACTER SET ascii COLLATE ascii_bin NULL,
  build_id CHAR(36) CHARACTER SET ascii COLLATE ascii_bin NULL,
  unit_id CHAR(36) CHARACTER SET ascii COLLATE ascii_bin NULL,
  evidence_id VARCHAR(32) NULL,
  source_type VARCHAR(16) NOT NULL DEFAULT 'DOCUMENT',
  tool_name VARCHAR(80) NULL,
  source_snapshot JSON NULL,
  purpose VARCHAR(16) NOT NULL,
  span_json JSON NULL,
  payload_hash BINARY(32) NULL,
  send_status VARCHAR(16) NOT NULL DEFAULT 'PREPARED',
  created_at DATETIME(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6),
  updated_at DATETIME(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6) ON UPDATE CURRENT_TIMESTAMP(6),
  PRIMARY KEY (tenant_id, id),
  KEY ix_source_document (tenant_id,document_id,run_id),
  KEY ix_source_run (tenant_id,run_id),
  CHECK ((source_type = 'DOCUMENT' AND document_id IS NOT NULL AND build_id IS NOT NULL AND source_snapshot IS NULL)
      OR (source_type = 'TOOL' AND document_id IS NULL AND build_id IS NULL AND source_snapshot IS NOT NULL AND tool_name IS NOT NULL)),
  CONSTRAINT fk_53 FOREIGN KEY (tenant_id) REFERENCES tenant (id),
  CONSTRAINT fk_54 FOREIGN KEY (tenant_id,run_id) REFERENCES qa_run (tenant_id,id),
  CONSTRAINT fk_55 FOREIGN KEY (tenant_id,document_id,build_id) REFERENCES index_build (tenant_id,document_id,id),
  CONSTRAINT fk_56 FOREIGN KEY (tenant_id,build_id,unit_id) REFERENCES source_unit (tenant_id,build_id,id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;

CREATE TABLE run_dependency (
  tenant_id CHAR(36) CHARACTER SET ascii COLLATE ascii_bin NOT NULL,
  id CHAR(36) CHARACTER SET ascii COLLATE ascii_bin NOT NULL,
  run_id CHAR(36) CHARACTER SET ascii COLLATE ascii_bin NOT NULL,
  prior_run_id CHAR(36) CHARACTER SET ascii COLLATE ascii_bin NOT NULL,
  created_at DATETIME(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6),
  updated_at DATETIME(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6) ON UPDATE CURRENT_TIMESTAMP(6),
  PRIMARY KEY (tenant_id, id),
  UNIQUE KEY uq_run_dependency (tenant_id,run_id,prior_run_id),
  CHECK (run_id <> prior_run_id),
  CONSTRAINT fk_57 FOREIGN KEY (tenant_id) REFERENCES tenant (id),
  CONSTRAINT fk_58 FOREIGN KEY (tenant_id,run_id) REFERENCES qa_run (tenant_id,id),
  CONSTRAINT fk_59 FOREIGN KEY (tenant_id,prior_run_id) REFERENCES qa_run (tenant_id,id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;

CREATE TABLE run_event (
  tenant_id CHAR(36) CHARACTER SET ascii COLLATE ascii_bin NOT NULL,
  id CHAR(36) CHARACTER SET ascii COLLATE ascii_bin NOT NULL,
  run_id CHAR(36) CHARACTER SET ascii COLLATE ascii_bin NOT NULL,
  sequence_no INT NOT NULL,
  stage VARCHAR(32) NOT NULL,
  duration_ms BIGINT NULL,
  payload JSON NULL,
  error_code VARCHAR(64) NULL,
  created_at DATETIME(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6),
  updated_at DATETIME(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6) ON UPDATE CURRENT_TIMESTAMP(6),
  PRIMARY KEY (tenant_id, id),
  UNIQUE KEY uq_event_sequence (tenant_id,run_id,sequence_no),
  CONSTRAINT fk_60 FOREIGN KEY (tenant_id) REFERENCES tenant (id),
  CONSTRAINT fk_61 FOREIGN KEY (tenant_id,run_id) REFERENCES qa_run (tenant_id,id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;

CREATE TABLE feedback (
  tenant_id CHAR(36) CHARACTER SET ascii COLLATE ascii_bin NOT NULL,
  id CHAR(36) CHARACTER SET ascii COLLATE ascii_bin NOT NULL,
  run_id CHAR(36) CHARACTER SET ascii COLLATE ascii_bin NOT NULL,
  submitted_by CHAR(36) CHARACTER SET ascii COLLATE ascii_bin NOT NULL,
  rating VARCHAR(8) NULL,
  issue_type VARCHAR(32) NULL,
  comment_text TEXT NULL,
  status VARCHAR(16) NOT NULL DEFAULT 'OPEN',
  resolution_text TEXT NULL,
  created_at DATETIME(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6),
  updated_at DATETIME(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6) ON UPDATE CURRENT_TIMESTAMP(6),
  PRIMARY KEY (tenant_id, id),
  UNIQUE KEY uq_feedback_run_user (tenant_id,run_id,submitted_by),
  CHECK (rating IS NULL OR rating IN ('USEFUL','USELESS')),
  CHECK (status IN ('OPEN','IN_PROGRESS','RESOLVED')),
  CONSTRAINT fk_62 FOREIGN KEY (tenant_id) REFERENCES tenant (id),
  CONSTRAINT fk_63 FOREIGN KEY (tenant_id,run_id) REFERENCES qa_run (tenant_id,id),
  CONSTRAINT fk_64 FOREIGN KEY (tenant_id,submitted_by) REFERENCES app_user (tenant_id,id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;

CREATE TABLE tenant_budget (
  tenant_id CHAR(36) CHARACTER SET ascii COLLATE ascii_bin NOT NULL,
  id CHAR(36) CHARACTER SET ascii COLLATE ascii_bin NOT NULL,
  period_month CHAR(7) CHARACTER SET ascii NOT NULL,
  currency CHAR(3) NOT NULL DEFAULT 'CNY',
  limit_amount DECIMAL(18,6) NOT NULL,
  reserved_amount DECIMAL(18,6) NOT NULL DEFAULT 0,
  settled_amount DECIMAL(18,6) NOT NULL DEFAULT 0,
  created_at DATETIME(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6),
  updated_at DATETIME(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6) ON UPDATE CURRENT_TIMESTAMP(6),
  PRIMARY KEY (tenant_id, id),
  UNIQUE KEY uq_budget_period (tenant_id,period_month),
  CHECK (limit_amount >= 0 AND reserved_amount >= 0 AND settled_amount >= 0),
  CONSTRAINT fk_65 FOREIGN KEY (tenant_id) REFERENCES tenant (id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;

CREATE TABLE budget_reservation (
  tenant_id CHAR(36) CHARACTER SET ascii COLLATE ascii_bin NOT NULL,
  id CHAR(36) CHARACTER SET ascii COLLATE ascii_bin NOT NULL,
  budget_id CHAR(36) CHARACTER SET ascii COLLATE ascii_bin NOT NULL,
  caller_environment VARCHAR(128) CHARACTER SET ascii COLLATE ascii_bin NOT NULL,
  caller_call_id CHAR(36) CHARACTER SET ascii COLLATE ascii_bin NOT NULL,
  status VARCHAR(16) NOT NULL,
  reserved_amount DECIMAL(18,6) NOT NULL,
  settled_amount DECIMAL(18,6) NULL,
  expires_at DATETIME(6) NOT NULL,
  created_at DATETIME(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6),
  updated_at DATETIME(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6) ON UPDATE CURRENT_TIMESTAMP(6),
  PRIMARY KEY (tenant_id,id),
  UNIQUE KEY uq_reservation_call (tenant_id,caller_environment,caller_call_id),
  KEY ix_reservation_status (tenant_id,status,expires_at),
  CHECK (reserved_amount >= 0),
  CONSTRAINT fk_206 FOREIGN KEY (tenant_id) REFERENCES tenant (id),
  CONSTRAINT fk_207 FOREIGN KEY (tenant_id,budget_id) REFERENCES tenant_budget (tenant_id,id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;

CREATE TABLE model_call (
  tenant_id CHAR(36) CHARACTER SET ascii COLLATE ascii_bin NOT NULL,
  id CHAR(36) CHARACTER SET ascii COLLATE ascii_bin NOT NULL,
  run_id CHAR(36) CHARACTER SET ascii COLLATE ascii_bin NULL,
  job_id CHAR(36) CHARACTER SET ascii COLLATE ascii_bin NULL,
  purpose VARCHAR(24) NOT NULL,
  model_config_id CHAR(36) CHARACTER SET ascii COLLATE ascii_bin NOT NULL,
  attempt_no INT NOT NULL,
  status VARCHAR(16) NOT NULL,
  reserved_amount DECIMAL(18,6) NOT NULL,
  settled_amount DECIMAL(18,6) NULL,
  input_tokens BIGINT NULL,
  output_tokens BIGINT NULL,
  provider_request_id VARCHAR(255) NULL,
  pricing_snapshot JSON NOT NULL,
  usage_detail JSON NULL,
  created_at DATETIME(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6),
  updated_at DATETIME(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6) ON UPDATE CURRENT_TIMESTAMP(6),
  reservation_authority VARCHAR(128) NOT NULL,
  reservation_id CHAR(36) CHARACTER SET ascii COLLATE ascii_bin NULL,
  PRIMARY KEY (tenant_id, id),
  KEY ix_call_run (tenant_id,run_id),
  KEY ix_call_job (tenant_id,job_id),
  CHECK ((run_id IS NULL) <> (job_id IS NULL)),
  CHECK (reserved_amount >= 0),
  CONSTRAINT fk_208 FOREIGN KEY (tenant_id) REFERENCES tenant (id),
  CONSTRAINT fk_209 FOREIGN KEY (tenant_id,run_id) REFERENCES qa_run (tenant_id,id),
  CONSTRAINT fk_210 FOREIGN KEY (tenant_id,job_id) REFERENCES job (tenant_id,id),
  CONSTRAINT fk_211 FOREIGN KEY (tenant_id,model_config_id) REFERENCES config_revision (tenant_id,id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;

CREATE TABLE eval_dataset_version (
  tenant_id CHAR(36) CHARACTER SET ascii COLLATE ascii_bin NOT NULL,
  id CHAR(36) CHARACTER SET ascii COLLATE ascii_bin NOT NULL,
  name VARCHAR(128) NOT NULL,
  revision INT NOT NULL,
  status VARCHAR(16) NOT NULL DEFAULT 'DRAFT',
  manifest_hash BINARY(32) NULL,
  frozen_at DATETIME(6) NULL,
  created_at DATETIME(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6),
  updated_at DATETIME(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6) ON UPDATE CURRENT_TIMESTAMP(6),
  PRIMARY KEY (tenant_id, id),
  UNIQUE KEY uq_dataset_revision (tenant_id,name,revision),
  CHECK (status IN ('DRAFT','FROZEN','PURGED')),
  CONSTRAINT fk_71 FOREIGN KEY (tenant_id) REFERENCES tenant (id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;

CREATE TABLE eval_case (
  tenant_id CHAR(36) CHARACTER SET ascii COLLATE ascii_bin NOT NULL,
  id CHAR(36) CHARACTER SET ascii COLLATE ascii_bin NOT NULL,
  dataset_version_id CHAR(36) CHARACTER SET ascii COLLATE ascii_bin NOT NULL,
  case_key VARCHAR(128) NOT NULL,
  split VARCHAR(8) NOT NULL,
  group_key VARCHAR(128) NOT NULL,
  category VARCHAR(32) NOT NULL,
  question_json JSON NOT NULL,
  persona_json JSON NOT NULL,
  expected_json JSON NOT NULL,
  annotation_status VARCHAR(16) NOT NULL,
  created_at DATETIME(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6),
  updated_at DATETIME(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6) ON UPDATE CURRENT_TIMESTAMP(6),
  PRIMARY KEY (tenant_id, id),
  UNIQUE KEY uq_case_key (tenant_id,dataset_version_id,case_key),
  UNIQUE KEY uq_case_dataset (tenant_id,dataset_version_id,id),
  CHECK (split IN ('DEV','TEST')),
  CONSTRAINT fk_72 FOREIGN KEY (tenant_id) REFERENCES tenant (id),
  CONSTRAINT fk_73 FOREIGN KEY (tenant_id,dataset_version_id) REFERENCES eval_dataset_version (tenant_id,id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;

CREATE TABLE eval_gold (
  tenant_id CHAR(36) CHARACTER SET ascii COLLATE ascii_bin NOT NULL,
  id CHAR(36) CHARACTER SET ascii COLLATE ascii_bin NOT NULL,
  case_id CHAR(36) CHARACTER SET ascii COLLATE ascii_bin NOT NULL,
  fact_group VARCHAR(128) NOT NULL,
  alternative_no INT NOT NULL,
  origin_system VARCHAR(128) NOT NULL,
  origin_tenant_id CHAR(36) CHARACTER SET ascii COLLATE ascii_bin NOT NULL,
  origin_document_id CHAR(36) CHARACTER SET ascii COLLATE ascii_bin NOT NULL,
  version_hash BINARY(32) NOT NULL,
  anchor_json JSON NOT NULL,
  relevance_grade INT NOT NULL,
  created_at DATETIME(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6),
  updated_at DATETIME(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6) ON UPDATE CURRENT_TIMESTAMP(6),
  PRIMARY KEY (tenant_id, id),
  CHECK (relevance_grade BETWEEN 0 AND 3),
  KEY ix_gold_origin (origin_system,origin_tenant_id,origin_document_id),
  CONSTRAINT fk_74 FOREIGN KEY (tenant_id) REFERENCES tenant (id),
  CONSTRAINT fk_75 FOREIGN KEY (tenant_id,case_id) REFERENCES eval_case (tenant_id,id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;

CREATE TABLE eval_snapshot (
  tenant_id CHAR(36) CHARACTER SET ascii COLLATE ascii_bin NOT NULL,
  id CHAR(36) CHARACTER SET ascii COLLATE ascii_bin NOT NULL,
  name VARCHAR(128) NOT NULL,
  manifest_hash BINARY(32) NOT NULL,
  manifest_json JSON NOT NULL,
  status VARCHAR(16) NOT NULL,
  environment_id VARCHAR(128) NOT NULL,
  created_at DATETIME(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6),
  updated_at DATETIME(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6) ON UPDATE CURRENT_TIMESTAMP(6),
  PRIMARY KEY (tenant_id, id),
  CONSTRAINT fk_76 FOREIGN KEY (tenant_id) REFERENCES tenant (id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;

CREATE TABLE eval_snapshot_document (
  tenant_id CHAR(36) CHARACTER SET ascii COLLATE ascii_bin NOT NULL,
  id CHAR(36) CHARACTER SET ascii COLLATE ascii_bin NOT NULL,
  snapshot_id CHAR(36) CHARACTER SET ascii COLLATE ascii_bin NOT NULL,
  document_id CHAR(36) CHARACTER SET ascii COLLATE ascii_bin NOT NULL,
  version_id CHAR(36) CHARACTER SET ascii COLLATE ascii_bin NOT NULL,
  origin_system VARCHAR(128) NOT NULL,
  origin_tenant_id CHAR(36) CHARACTER SET ascii COLLATE ascii_bin NOT NULL,
  origin_document_id CHAR(36) CHARACTER SET ascii COLLATE ascii_bin NOT NULL,
  origin_version_hash BINARY(32) NOT NULL,
  created_at DATETIME(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6),
  updated_at DATETIME(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6) ON UPDATE CURRENT_TIMESTAMP(6),
  PRIMARY KEY (tenant_id, id),
  UNIQUE KEY uq_snapshot_doc (tenant_id,snapshot_id,document_id),
  KEY ix_snapshot_origin (origin_system,origin_tenant_id,origin_document_id),
  CONSTRAINT fk_77 FOREIGN KEY (tenant_id) REFERENCES tenant (id),
  CONSTRAINT fk_78 FOREIGN KEY (tenant_id,snapshot_id) REFERENCES eval_snapshot (tenant_id,id),
  CONSTRAINT fk_79 FOREIGN KEY (tenant_id,document_id,version_id) REFERENCES document_version (tenant_id,document_id,id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;

CREATE TABLE eval_run (
  tenant_id CHAR(36) CHARACTER SET ascii COLLATE ascii_bin NOT NULL,
  id CHAR(36) CHARACTER SET ascii COLLATE ascii_bin NOT NULL,
  dataset_version_id CHAR(36) CHARACTER SET ascii COLLATE ascii_bin NOT NULL,
  snapshot_id CHAR(36) CHARACTER SET ascii COLLATE ascii_bin NOT NULL,
  config_id CHAR(36) CHARACTER SET ascii COLLATE ascii_bin NOT NULL,
  job_id CHAR(36) CHARACTER SET ascii COLLATE ascii_bin NOT NULL,
  status VARCHAR(24) NOT NULL,
  manifest_json JSON NOT NULL,
  metrics_json JSON NULL,
  created_at DATETIME(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6),
  updated_at DATETIME(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6) ON UPDATE CURRENT_TIMESTAMP(6),
  PRIMARY KEY (tenant_id, id),
  UNIQUE KEY uq_eval_run_dataset (tenant_id,dataset_version_id,id),
  CONSTRAINT fk_80 FOREIGN KEY (tenant_id) REFERENCES tenant (id),
  CONSTRAINT fk_81 FOREIGN KEY (tenant_id,dataset_version_id) REFERENCES eval_dataset_version (tenant_id,id),
  CONSTRAINT fk_82 FOREIGN KEY (tenant_id,snapshot_id) REFERENCES eval_snapshot (tenant_id,id),
  CONSTRAINT fk_83 FOREIGN KEY (tenant_id,config_id) REFERENCES config_revision (tenant_id,id),
  CONSTRAINT fk_84 FOREIGN KEY (tenant_id,job_id) REFERENCES job (tenant_id,id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;

CREATE TABLE eval_result (
  tenant_id CHAR(36) CHARACTER SET ascii COLLATE ascii_bin NOT NULL,
  id CHAR(36) CHARACTER SET ascii COLLATE ascii_bin NOT NULL,
  eval_run_id CHAR(36) CHARACTER SET ascii COLLATE ascii_bin NOT NULL,
  dataset_version_id CHAR(36) CHARACTER SET ascii COLLATE ascii_bin NOT NULL,
  case_id CHAR(36) CHARACTER SET ascii COLLATE ascii_bin NOT NULL,
  repeat_no INT NOT NULL DEFAULT 0,
  qa_run_id CHAR(36) CHARACTER SET ascii COLLATE ascii_bin NULL,
  status VARCHAR(24) NOT NULL,
  metrics_json JSON NULL,
  judge_json JSON NULL,
  error_code VARCHAR(64) NULL,
  created_at DATETIME(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6),
  updated_at DATETIME(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6) ON UPDATE CURRENT_TIMESTAMP(6),
  PRIMARY KEY (tenant_id, id),
  UNIQUE KEY uq_eval_result (tenant_id,eval_run_id,case_id,repeat_no),
  CONSTRAINT fk_85 FOREIGN KEY (tenant_id) REFERENCES tenant (id),
  CONSTRAINT fk_86 FOREIGN KEY (tenant_id,dataset_version_id,eval_run_id) REFERENCES eval_run (tenant_id,dataset_version_id,id),
  CONSTRAINT fk_87 FOREIGN KEY (tenant_id,dataset_version_id,case_id) REFERENCES eval_case (tenant_id,dataset_version_id,id),
  CONSTRAINT fk_88 FOREIGN KEY (tenant_id,qa_run_id) REFERENCES qa_run (tenant_id,id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;

CREATE TABLE eval_review (
  tenant_id CHAR(36) CHARACTER SET ascii COLLATE ascii_bin NOT NULL,
  id CHAR(36) CHARACTER SET ascii COLLATE ascii_bin NOT NULL,
  result_id CHAR(36) CHARACTER SET ascii COLLATE ascii_bin NOT NULL,
  reviewer_id CHAR(36) CHARACTER SET ascii COLLATE ascii_bin NOT NULL,
  labels_json JSON NOT NULL,
  reason_text TEXT NULL,
  created_at DATETIME(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6),
  updated_at DATETIME(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6) ON UPDATE CURRENT_TIMESTAMP(6),
  PRIMARY KEY (tenant_id, id),
  CONSTRAINT fk_89 FOREIGN KEY (tenant_id) REFERENCES tenant (id),
  CONSTRAINT fk_90 FOREIGN KEY (tenant_id,result_id) REFERENCES eval_result (tenant_id,id),
  CONSTRAINT fk_91 FOREIGN KEY (tenant_id,reviewer_id) REFERENCES app_user (tenant_id,id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;

CREATE TABLE audit_event (
  tenant_id CHAR(36) CHARACTER SET ascii COLLATE ascii_bin NOT NULL,
  id CHAR(36) CHARACTER SET ascii COLLATE ascii_bin NOT NULL,
  actor_id CHAR(36) CHARACTER SET ascii COLLATE ascii_bin NULL,
  action VARCHAR(64) NOT NULL,
  object_type VARCHAR(32) NOT NULL,
  object_id CHAR(36) CHARACTER SET ascii COLLATE ascii_bin NOT NULL,
  outcome VARCHAR(16) NOT NULL,
  metadata_json JSON NULL,
  created_at DATETIME(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6),
  updated_at DATETIME(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6) ON UPDATE CURRENT_TIMESTAMP(6),
  PRIMARY KEY (tenant_id, id),
  KEY ix_audit_object (tenant_id,object_type,object_id,created_at),
  CONSTRAINT fk_92 FOREIGN KEY (tenant_id) REFERENCES tenant (id),
  CONSTRAINT fk_93 FOREIGN KEY (tenant_id,actor_id) REFERENCES app_user (tenant_id,id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;

CREATE TABLE deletion_record (
  tenant_id CHAR(36) CHARACTER SET ascii COLLATE ascii_bin NOT NULL,
  id CHAR(36) CHARACTER SET ascii COLLATE ascii_bin NOT NULL,
  document_id CHAR(36) CHARACTER SET ascii COLLATE ascii_bin NOT NULL,
  origin_system VARCHAR(128) NOT NULL,
  requested_by CHAR(36) CHARACTER SET ascii COLLATE ascii_bin NOT NULL,
  status VARCHAR(24) NOT NULL,
  domains_json JSON NOT NULL,
  backup_expiry_at DATETIME(6) NOT NULL,
  completed_at DATETIME(6) NULL,
  created_at DATETIME(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6),
  updated_at DATETIME(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6) ON UPDATE CURRENT_TIMESTAMP(6),
  PRIMARY KEY (tenant_id, id),
  KEY ix_deletion_doc (tenant_id,document_id),
  CONSTRAINT fk_94 FOREIGN KEY (tenant_id) REFERENCES tenant (id),
  CONSTRAINT fk_95 FOREIGN KEY (tenant_id,document_id) REFERENCES document (tenant_id,id),
  CONSTRAINT fk_96 FOREIGN KEY (tenant_id,requested_by) REFERENCES app_user (tenant_id,id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;

-- 这是延迟校验的循环引用：document.active_build_id 指向的 build 必须属于该 document；发布流程在事务中保证这一不变量。
ALTER TABLE document ADD CONSTRAINT fk_document_active_build
  FOREIGN KEY (tenant_id,id,active_build_id)
  REFERENCES index_build (tenant_id,document_id,id);

ALTER TABLE tenant ADD CONSTRAINT fk_tenant_retrieval_config
  FOREIGN KEY (id,active_retrieval_config_id)
  REFERENCES config_revision (tenant_id,id);

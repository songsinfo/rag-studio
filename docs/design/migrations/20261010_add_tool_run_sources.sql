-- 部署新后端前执行；旧 DOCUMENT 行原值不变。
-- 注意：必须先备份；回滚旧代码前不得删除 TOOL 来源或将 document_id 恢复 NOT NULL。
ALTER TABLE run_source
  MODIFY document_id CHAR(36) CHARACTER SET ascii COLLATE ascii_bin NULL,
  MODIFY build_id CHAR(36) CHARACTER SET ascii COLLATE ascii_bin NULL,
  ADD COLUMN source_type VARCHAR(16) NOT NULL DEFAULT 'DOCUMENT' AFTER evidence_id,
  ADD COLUMN tool_name VARCHAR(80) NULL AFTER source_type,
  ADD COLUMN source_snapshot JSON NULL AFTER tool_name;
-- MySQL 8.4 对 NULL 的复合外键允许 TOOL 无 document/build。
ALTER TABLE run_source ADD CONSTRAINT chk_run_source_kind
 CHECK ((source_type = 'DOCUMENT' AND document_id IS NOT NULL AND build_id IS NOT NULL AND source_snapshot IS NULL)
     OR (source_type = 'TOOL' AND document_id IS NULL AND build_id IS NULL
         AND tool_name IS NOT NULL AND source_snapshot IS NOT NULL));

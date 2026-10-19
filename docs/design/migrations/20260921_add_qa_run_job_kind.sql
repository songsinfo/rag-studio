-- 升级现有 MySQL 8.x 数据库，使 job.kind 允许持久化 QA_RUN 任务；用于把 QA 执行从进程内任务迁移到 durable worker。
-- 不假设 MySQL 自动生成的 CHECK 名称，先从元数据中发现旧 job.kind 约束，再安全替换，避免不同环境约束名不一致。
SET @schema_name = DATABASE();
SET @old_job_kind_check = (
  SELECT tc.CONSTRAINT_NAME
  FROM information_schema.TABLE_CONSTRAINTS AS tc
  JOIN information_schema.CHECK_CONSTRAINTS AS cc
    ON cc.CONSTRAINT_SCHEMA = tc.CONSTRAINT_SCHEMA
   AND cc.CONSTRAINT_NAME = tc.CONSTRAINT_NAME
  WHERE tc.CONSTRAINT_SCHEMA = @schema_name
    AND tc.TABLE_NAME = 'job'
    AND tc.CONSTRAINT_TYPE = 'CHECK'
    AND cc.CHECK_CLAUSE LIKE '%kind%'
    AND cc.CHECK_CLAUSE LIKE '%INGEST%'
  LIMIT 1
);
SET @drop_job_kind_check_sql = IF(
  @old_job_kind_check IS NULL,
  'SELECT 1',
  CONCAT('ALTER TABLE job DROP CHECK `', REPLACE(@old_job_kind_check, '`', '``'), '`')
);
PREPARE drop_job_kind_check_stmt FROM @drop_job_kind_check_sql;
EXECUTE drop_job_kind_check_stmt;
DEALLOCATE PREPARE drop_job_kind_check_stmt;

ALTER TABLE job
  ADD CONSTRAINT chk_job_kind
  CHECK (kind IN ('INGEST','PURGE','RESTORE','EVALUATE','EXPORT','CLEANUP','QA_RUN'));

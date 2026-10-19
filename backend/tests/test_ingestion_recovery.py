import pytest
from sqlalchemy import create_engine, text

from app.errors import ApiException
from app.knowledge.service import KnowledgeService


def test_reset_unpublished_build_respects_parent_child_foreign_key():
    engine = create_engine("sqlite:///:memory:")
    with engine.begin() as conn:
        conn.exec_driver_sql("PRAGMA foreign_keys=ON")
        # RESTRICT 模拟 MySQL 对自引用外键的立即检查。
        for ddl in [
            "CREATE TABLE document (tenant_id TEXT, id TEXT, active_build_id TEXT)",
            "CREATE TABLE index_build (tenant_id TEXT, id TEXT, document_id TEXT, status TEXT)",
            "CREATE TABLE chunk (tenant_id TEXT, id TEXT, build_id TEXT, parent_id TEXT, PRIMARY KEY (tenant_id, build_id, id), FOREIGN KEY (tenant_id, build_id, parent_id) REFERENCES chunk (tenant_id, build_id, id) ON DELETE RESTRICT)",
            "CREATE TABLE representation (tenant_id TEXT, build_id TEXT)",
            "CREATE TABLE chunk_span (tenant_id TEXT, build_id TEXT)",
            "CREATE TABLE source_unit (tenant_id TEXT, build_id TEXT)",
        ]:
            conn.exec_driver_sql(ddl)
        conn.exec_driver_sql("INSERT INTO document VALUES ('tenant', 'doc', NULL)")
        conn.exec_driver_sql("INSERT INTO index_build VALUES ('tenant', 'build', 'doc', 'READY')")
        conn.exec_driver_sql("INSERT INTO chunk VALUES ('tenant', 'parent', 'build', NULL)")
        conn.exec_driver_sql("INSERT INTO chunk VALUES ('tenant', 'child', 'build', 'parent')")
        conn.exec_driver_sql("INSERT INTO chunk VALUES ('other', 'keep', 'build', NULL)")

        with pytest.raises(ApiException) as exc:
            KnowledgeService._reset_unpublished_build(conn, "tenant", "doc", "build")
        assert exc.value.code == "BUILD_ALREADY_PUBLISHED"
        conn.exec_driver_sql("UPDATE index_build SET status='QUEUED'")

        KnowledgeService._reset_unpublished_build(conn, "tenant", "doc", "build")
        KnowledgeService._reset_unpublished_build(conn, "tenant", "doc", "build")
        assert conn.execute(text("SELECT tenant_id, id FROM chunk")).all() == [("other", "keep")]
    engine.dispose()

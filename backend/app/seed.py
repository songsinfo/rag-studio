import hashlib
import json
import logging
import os
import uuid
from datetime import datetime, timezone
import bcrypt
from sqlalchemy import select

from app.db import get_db_connection, engine
from app.models import (
    metadata,
    tenant_table,
    app_user_table,
    user_group_table,
    group_member_table,
    config_revision_table,
)

logger = logging.getLogger(__name__)


class DemoSeed:
    @staticmethod
    def run():
        """确保数据表存在，并幂等补齐本地演示所需的租户、管理员、运维组和默认配置；重复执行不会覆盖已有业务数据。"""
        seed_pwd = os.getenv("RAG_SEED_PASSWORD", "").strip()
        if not seed_pwd:
            raise RuntimeError("RAG_SEED_PASSWORD is required; initialize the environment first")
        now = datetime.now(timezone.utc)

        # 第 1 步：确保元数据表已创建；seed 本身保持幂等，重复启动不会重复建表或破坏现有数据。
        try:
            metadata.create_all(bind=engine)
        except Exception as e:
            logger.warning("metadata.create_all check: %s", e)

        with get_db_connection() as conn:
            # 第 2 步：写入演示租户 tenant-demo-1；仅在不存在时创建，已有租户配置不被覆盖。
            tenant = conn.execute(
                select(tenant_table).where(tenant_table.c.code == "demo")
            ).first()

            tenant_id = "tenant-demo-1"
            if not tenant:
                logger.info("Seeding default demo tenant: %s", tenant_id)
                conn.execute(
                    tenant_table.insert().values(
                        id=tenant_id,
                        code="demo",
                        name="Demo Tenant",
                        security_epoch=1,
                        active_retrieval_config_id="cfg-default-retrieval",
                        created_at=now,
                    )
                )
                conn.commit()

            else:
                tenant_id = str(tenant.id)

            # 第 3 步：创建演示管理员用户，供本地联调和端到端测试使用。
            admin_user = conn.execute(
                select(app_user_table).where(
                    app_user_table.c.tenant_id == tenant_id,
                    app_user_table.c.login_name == "admin",
                )
            ).first()

            user_id = "user-admin-1"
            if not admin_user:
                hashed_pwd = bcrypt.hashpw(seed_pwd.encode("utf-8"), bcrypt.gensalt(rounds=10)).decode("utf-8")
                logger.info("Seeding default admin user: admin")
                conn.execute(
                    app_user_table.insert().values(
                        tenant_id=tenant_id,
                        id=user_id,
                        login_name="admin",
                        password_hash=hashed_pwd,
                        is_admin=True,
                        status="ACTIVE",
                        created_at=now,
                        updated_at=now,
                    )
                )
                conn.commit()

            else:
                user_id = str(admin_user.id)

            # 第 4 步：创建默认运维组，用于验证组成员与 ACL 继承逻辑。
            ops_group = conn.execute(
                select(user_group_table).where(
                    user_group_table.c.tenant_id == tenant_id,
                    user_group_table.c.name == "Ops Group",
                )
            ).first()

            group_id = "group-ops-1"
            if not ops_group:
                logger.info("Seeding default user group: Ops Group")
                conn.execute(
                    user_group_table.insert().values(
                        tenant_id=tenant_id,
                        id=group_id,
                        name="Ops Group",
                        created_at=now,
                        updated_at=now,
                    )
                )
                conn.commit()

            else:
                group_id = str(ops_group.id)

            # 第 5 步：建立演示用户与运维组的成员关系，保证权限链路有可测试数据。
            gm = conn.execute(
                select(group_member_table).where(
                    group_member_table.c.tenant_id == tenant_id,
                    group_member_table.c.group_id == group_id,
                    group_member_table.c.user_id == user_id,
                )
            ).first()
            if not gm:
                conn.execute(
                    group_member_table.insert().values(
                        tenant_id=tenant_id,
                        id=f"gm-{uuid.uuid4().hex[:8]}",
                        group_id=group_id,
                        user_id=user_id,
                        created_at=now,
                        updated_at=now,
                    )
                )
                conn.commit()

            # 第 6 步：写入版本化企业检索策略；配置版本必须固定，历史运行才能按原参数复现。
            enterprise_cfg_id = "cfg-enterprise-retrieval-v2"
            enterprise_cfg = conn.execute(
                select(config_revision_table).where(
                    config_revision_table.c.tenant_id == tenant_id,
                    config_revision_table.c.id == enterprise_cfg_id,
                )
            ).first()
            if not enterprise_cfg:
                enterprise_payload = {
                    "retrieval": {
                        "mode": "HYBRID",
                        "denseTopK": 30,
                        "sparseTopK": 30,
                        "fusionTopK": 20,
                        "rrfK": 60,
                        "finalTopK": 8,
                    },
                    "rerank": {
                        "enabled": True,
                        "topK": 10,
                        "model": "rerank-main",
                        "failOpen": True,
                    },
                    "parentExpansion": {
                        "enabled": True,
                        "maxChars": 6000,
                    },
                }
                payload_str = json.dumps(enterprise_payload, sort_keys=True)
                conn.execute(
                    config_revision_table.insert().values(
                        tenant_id=tenant_id,
                        id=enterprise_cfg_id,
                        kind="RETRIEVAL",
                        name="enterprise_retrieval_pipeline",
                        revision=2,
                        sha256=hashlib.sha256(payload_str.encode("utf-8")).digest(),
                        payload=enterprise_payload,
                        status="PUBLISHED",
                        created_at=now,
                        updated_at=now,
                    )
                )
                conn.commit()

            tenant_cfg = conn.execute(
                select(tenant_table.c.active_retrieval_config_id).where(
                    tenant_table.c.id == tenant_id
                )
            ).scalar()
            if not tenant_cfg or str(tenant_cfg) == "cfg-default-retrieval":
                conn.execute(
                    tenant_table.update()
                    .where(tenant_table.c.id == tenant_id)
                    .values(active_retrieval_config_id=enterprise_cfg_id)
                )
                conn.commit()

            # 保留 v1 配置供历史 run 回放和对比，不能因为默认策略升级就修改已完成实验的解释上下文。
            # 第 6b 步：设置默认检索策略配置，新请求使用当前版本，历史请求仍引用自己的已发布版本。
            default_cfg = conn.execute(
                select(config_revision_table).where(
                    config_revision_table.c.tenant_id == tenant_id,
                    config_revision_table.c.id == "cfg-default-retrieval",
                )
            ).first()
            if not default_cfg:
                payload = {
                    "retrieval": {
                        "mode": "HYBRID",
                        "denseWeight": 0.5,
                        "bm25Weight": 0.5,
                        "topK": 10,
                        "rrfK": 60,
                    }
                }
                payload_str = json.dumps(payload, sort_keys=True)
                sha256_bytes = hashlib.sha256(payload_str.encode("utf-8")).digest()
                conn.execute(
                    config_revision_table.insert().values(
                        tenant_id=tenant_id,
                        id="cfg-default-retrieval",
                        kind="RETRIEVAL",
                        name="default_retrieval_strategy",
                        revision=1,
                        sha256=sha256_bytes,
                        payload=payload,
                        status="PUBLISHED",
                        created_at=now,
                        updated_at=now,
                    )
                )
                conn.commit()

            # 第 7 步：写入 model_call 记录会引用的模型网关别名，确保 usage/成本审计能映射到稳定模型标识。
            model_configs = {
                "cfg-model-chat-main": "chat-main",
                "cfg-model-embedding-main": "embedding-main",
                "cfg-model-judge-main": "judge-main",
                "cfg-model-rerank-main": "rerank-main",
            }
            for config_id, alias in model_configs.items():
                exists_row = conn.execute(
                    select(config_revision_table.c.id, config_revision_table.c.payload).where(
                        config_revision_table.c.tenant_id == tenant_id,
                        config_revision_table.c.id == config_id,
                    )
                ).first()
                payload = {
                    "modelAlias": alias,
                    "currency": "CNY",
                    "inputPricePerMillion": 0,
                    "outputPricePerMillion": 0,
                }
                if exists_row:
                    current_payload = dict(exists_row.payload or {})
                    if (
                        "inputPricePerMillion" not in current_payload
                        or "outputPricePerMillion" not in current_payload
                        or str(current_payload.get("currency", "CNY")).upper() != "CNY"
                    ):
                        merged = {**current_payload, **payload}
                        payload_str = json.dumps(merged, sort_keys=True)
                        conn.execute(
                            config_revision_table.update()
                            .where(
                                config_revision_table.c.tenant_id == tenant_id,
                                config_revision_table.c.id == config_id,
                            )
                            .values(
                                payload=merged,
                                sha256=hashlib.sha256(payload_str.encode("utf-8")).digest(),
                                updated_at=now,
                            )
                        )
                    continue
                conn.execute(
                    config_revision_table.insert().values(
                        tenant_id=tenant_id,
                        id=config_id,
                        kind="MODEL",
                        name=alias,
                        revision=1,
                        sha256=hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).digest(),
                        payload=payload,
                        status="PUBLISHED",
                        created_at=now,
                        updated_at=now,
                    )
                )
            conn.commit()

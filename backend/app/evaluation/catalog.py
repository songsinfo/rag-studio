import hashlib
import json
import uuid
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional, Set

from sqlalchemy import select, and_, or_, func, text, update, delete
from app.db import get_db_connection
from app.errors import ApiException
from app.identity.deps import Actor, can_read_document
from app.config import settings
from app.models import (
    tenant_table,
    document_table,
    document_version_table,
    index_build_table,
    source_unit_table,
    eval_dataset_version_table,
    eval_case_table,
    eval_gold_table,
    eval_snapshot_table,
    eval_snapshot_document_table,
)


def hash_payload(v: Any) -> bytes:
    """对确定性 JSON 表示计算 SHA-256 摘要；字段排序稳定后，相同输入可得到稳定哈希。"""
    serialized = json.dumps(v, sort_keys=True, separators=(',', ':'), ensure_ascii=False)
    return hashlib.sha256(serialized.encode("utf-8")).digest()


def obj(v: Any) -> Dict[str, Any]:
    """将 None/空字符串转为空字典，保留 dict；字符串/字节按 JSON 解析，其他输入通过 dict(v) 转换，转换失败时异常继续抛出。"""
    if v is None:
        return {}
    if isinstance(v, dict):
        return v
    if isinstance(v, (bytes, bytearray)):
        return json.loads(v.decode("utf-8"))
    if isinstance(v, str):
        return json.loads(v) if v.strip() else {}
    return dict(v)


def maps(v: Any) -> List[Dict[str, Any]]:
    """None 返回空列表；其他输入必须是 list，并逐项调用 obj 转换，不额外过滤元素，转换失败时异常继续抛出。"""
    if v is None:
        return []
    if not isinstance(v, list):
        raise ApiException(400, "INVALID_INPUT", "需要对象数组")
    return [obj(item) for item in v]


def dataset_view(r: Any) -> Dict[str, Any]:
    """把题集版本数据库记录转换为客户端响应结构，统一字段命名并隐藏内部存储细节。"""
    frozen_at = r.frozen_at.isoformat() if getattr(r, "frozen_at", None) else None
    return {
        "id": str(r.id),
        "name": str(r.name),
        "revision": int(r.revision),
        "status": str(r.status),
        "frozenAt": frozen_at,
    }


def case_view(r: Any) -> Dict[str, Any]:
    """把评估 case、预期答案类型与 Gold 信息整理为客户端可消费的响应结构。"""
    return {
        "id": str(r.id),
        "caseKey": str(r.case_key),
        "groupKey": str(r.group_key),
        "split": str(r.split),
        "category": str(r.category),
        "question": obj(r.question_json),
        "persona": obj(r.persona_json),
        "expected": obj(r.expected_json),
        "annotationStatus": str(r.annotation_status),
    }


class EvaluationCatalog:
    """评估题集版本、case、Gold 事实与语料快照的统一管理服务；负责冻结、复制、校验和可复现性约束。"""

    @staticmethod
    def require_admin(actor: Actor):
        if not actor.admin:
            raise ApiException(403, "ACCESS_DENIED", "需要管理员权限")
        env = getattr(settings, "RAG_ENVIRONMENT", "demo").lower()
        if env not in {"demo", "test", "evaluation", "development"}:
            raise ApiException(409, "EVALUATION_ENVIRONMENT_REQUIRED", "请在独立 evaluation 实例运行评估，不能使用工作库")

    @staticmethod
    def datasets(actor: Actor) -> List[Dict[str, Any]]:
        EvaluationCatalog.require_admin(actor)
        with get_db_connection() as conn:
            stmt = (
                select(eval_dataset_version_table)
                .where(eval_dataset_version_table.c.tenant_id == actor.tenant_id)
                .order_by(eval_dataset_version_table.c.created_at.desc())
            )
            rows = conn.execute(stmt).fetchall()
            return [dataset_view(r) for r in rows]

    @staticmethod
    def create_dataset(actor: Actor, body: Dict[str, Any]) -> Dict[str, Any]:
        EvaluationCatalog.require_admin(actor)
        name = str(body.get("name", "")).strip()
        if not name:
            raise ApiException(400, "INVALID_INPUT", "name 必填")

        now = datetime.now(timezone.utc)
        dataset_id = str(uuid.uuid4())

        with get_db_connection() as conn:
            max_rev_stmt = select(func.coalesce(func.max(eval_dataset_version_table.c.revision), 0) + 1).where(
                and_(
                    eval_dataset_version_table.c.tenant_id == actor.tenant_id,
                    eval_dataset_version_table.c.name == name,
                )
            )
            next_rev = conn.execute(max_rev_stmt).scalar() or 1

            conn.execute(
                eval_dataset_version_table.insert().values(
                    tenant_id=actor.tenant_id,
                    id=dataset_id,
                    name=name,
                    revision=next_rev,
                    status="DRAFT",
                    manifest_hash=None,
                    frozen_at=None,
                    created_at=now,
                    updated_at=now,
                )
            )
            conn.commit()

            row = conn.execute(
                select(eval_dataset_version_table).where(
                    and_(
                        eval_dataset_version_table.c.tenant_id == actor.tenant_id,
                        eval_dataset_version_table.c.id == dataset_id,
                    )
                )
            ).first()
            return dataset_view(row)

    @staticmethod
    def copy_version(actor: Actor, source_id: str) -> Dict[str, Any]:
        EvaluationCatalog.require_admin(actor)
        with get_db_connection() as conn:
            source = conn.execute(
                select(eval_dataset_version_table).where(
                    and_(
                        eval_dataset_version_table.c.tenant_id == actor.tenant_id,
                        eval_dataset_version_table.c.id == source_id,
                    )
                )
            ).first()
            if not source:
                raise ApiException(404, "RESOURCE_NOT_FOUND", "源题集版本不存在")

        target = EvaluationCatalog.create_dataset(actor, {"name": source.name})
        cases_list = EvaluationCatalog.cases(actor, source_id)
        if cases_list:
            EvaluationCatalog.save_cases(actor, target["id"], cases_list)
        return target

    @staticmethod
    def _require_draft(conn, actor: Actor, version_id: str):
        stmt = select(eval_dataset_version_table.c.status).where(
            and_(
                eval_dataset_version_table.c.tenant_id == actor.tenant_id,
                eval_dataset_version_table.c.id == version_id,
            )
        )
        row = conn.execute(stmt).first()
        if not row:
            raise ApiException(404, "RESOURCE_NOT_FOUND", "题集版本不存在")
        if row.status != "DRAFT":
            raise ApiException(409, "DATASET_FROZEN", "冻结题集必须创建新版本再修改")

    @staticmethod
    def cases(actor: Actor, version_id: str) -> List[Dict[str, Any]]:
        EvaluationCatalog.require_admin(actor)
        with get_db_connection() as conn:
            ver = conn.execute(
                select(eval_dataset_version_table.c.id).where(
                    and_(
                        eval_dataset_version_table.c.tenant_id == actor.tenant_id,
                        eval_dataset_version_table.c.id == version_id,
                    )
                )
            ).first()
            if not ver:
                raise ApiException(404, "RESOURCE_NOT_FOUND", "题集版本不存在")

            stmt = (
                select(eval_case_table)
                .where(
                    and_(
                        eval_case_table.c.tenant_id == actor.tenant_id,
                        eval_case_table.c.dataset_version_id == version_id,
                    )
                )
                .order_by(eval_case_table.c.case_key.asc())
            )
            rows = conn.execute(stmt).fetchall()

            res = []
            for r in rows:
                c_dict = case_view(r)
                # 校验 Gold evidence 所引用的文档在当前租户/快照下确实可读；不可见文档不能进入 Gold，避免评估集泄漏越权内容。
                expected = c_dict.get("expected", {})
                for grp in expected.get("evidenceGroups", []):
                    for alt in grp.get("alternatives", []):
                        doc_id = alt.get("documentId")
                        if doc_id and not can_read_document(conn, actor, str(doc_id)):
                            raise ApiException(403, "ACCESS_DENIED", f"文档不可见: {doc_id}")
                res.append(c_dict)
            return res

    @staticmethod
    def save_cases(actor: Actor, version_id: str, inputs: List[Dict[str, Any]]) -> Dict[str, Any]:
        EvaluationCatalog.require_admin(actor)
        if not inputs or len(inputs) > 1000:
            raise ApiException(400, "INVALID_INPUT", "一次需包含1到1000题")

        now = datetime.now(timezone.utc)
        env = getattr(settings, "RAG_ENVIRONMENT", "demo")

        with get_db_connection() as conn:
            EvaluationCatalog._require_draft(conn, actor, version_id)

            for b in inputs:
                case_key = str(b.get("caseKey", "")).strip()
                if not case_key:
                    raise ApiException(400, "INVALID_INPUT", "caseKey 必填")

                split = str(b.get("split", "")).strip().upper()
                if split not in {"DEV", "TEST"}:
                    raise ApiException(400, "INVALID_INPUT", "split 必须 DEV/TEST")

                group_key = str(b.get("groupKey", "")).strip()
                if not group_key:
                    raise ApiException(400, "INVALID_INPUT", "groupKey 必填")

                category = str(b.get("category", "")).strip()
                if not category:
                    raise ApiException(400, "INVALID_INPUT", "category 必填")

                q_raw = b.get("question")
                question = {"text": q_raw} if isinstance(q_raw, str) else obj(q_raw)
                if not str(question.get("text", "")).strip() and not maps(question.get("turns")):
                    raise ApiException(400, "INVALID_INPUT", "问题不能为空")

                persona = obj(b.get("persona"))
                if not str(persona.get("loginName", "")).strip():
                    raise ApiException(400, "INVALID_INPUT", "loginName 必填")

                expected = obj(b.get("expected"))
                exp_status = str(expected.get("status", "")).strip().upper()
                if exp_status not in {"ANSWER", "PARTIAL", "ABSTAIN", "CONFLICT"}:
                    raise ApiException(400, "INVALID_INPUT", "预期状态无效")

                annotation = str(b.get("annotationStatus", "")).strip() or "DRAFT"
                if annotation not in {"DRAFT", "AUTO_REVIEWED", "HUMAN_REVIEWED"}:
                    raise ApiException(400, "INVALID_INPUT", "标注状态无效")

                # 冻结前同时校验锚点和文档存在性，把坏引用提前拦截，避免运行阶段才发现 Gold 无法解析。
                for grp in maps(expected.get("evidenceGroups")):
                    for alt in maps(grp.get("alternatives")):
                        doc_id = str(alt.get("documentId", "")).strip()
                        ver_id = str(alt.get("versionId", "")).strip()
                        quote = str(alt.get("quote", "")).strip()
                        if not doc_id or not ver_id or not quote:
                            raise ApiException(400, "INVALID_INPUT", "证据锚点字段不完整")
                        if not can_read_document(conn, actor, doc_id):
                            raise ApiException(403, "ACCESS_DENIED", f"文档不可见: {doc_id}")

                # 按当前租户、题集版本和 caseKey 查找已有 case 并复用其 ID；不存在时生成新 ID，供 Gold anchors 关联。
                existing_case = conn.execute(
                    select(eval_case_table.c.id).where(
                        and_(
                            eval_case_table.c.tenant_id == actor.tenant_id,
                            eval_case_table.c.dataset_version_id == version_id,
                            eval_case_table.c.case_key == case_key,
                        )
                    )
                ).first()

                if existing_case:
                    case_id = str(existing_case.id)
                    conn.execute(
                        delete(eval_gold_table).where(
                            and_(
                                eval_gold_table.c.tenant_id == actor.tenant_id,
                                eval_gold_table.c.case_id == case_id,
                            )
                        )
                    )
                    conn.execute(
                        delete(eval_case_table).where(
                            and_(
                                eval_case_table.c.tenant_id == actor.tenant_id,
                                eval_case_table.c.id == case_id,
                            )
                        )
                    )
                else:
                    case_id = str(uuid.uuid4())

                # 先写入 case 主记录，再写 Gold anchors；这样外键关系和后续校验都有明确父记录。
                conn.execute(
                    eval_case_table.insert().values(
                        tenant_id=actor.tenant_id,
                        id=case_id,
                        dataset_version_id=version_id,
                        case_key=case_key,
                        split=split,
                        group_key=group_key,
                        category=category,
                        question_json=question,
                        persona_json=persona,
                        expected_json=expected,
                        annotation_status=annotation,
                        created_at=now,
                        updated_at=now,
                    )
                )

                # 写入 Gold anchors，并保留文档、版本和位置等定位信息，供检索命中与答案归因评估使用。
                for grp in maps(expected.get("evidenceGroups")):
                    fact_name = str(grp.get("fact", "")).strip()
                    for alt_no, alt in enumerate(maps(grp.get("alternatives"))):
                        doc_id = str(alt.get("documentId", "")).strip()
                        ver_id = str(alt.get("versionId", "")).strip()
                        grade = int(alt.get("relevanceGrade", 3))
                        if grade < 0 or grade > 3:
                            raise ApiException(400, "INVALID_INPUT", "相关等级必须0..3")

                        doc_ver = conn.execute(
                            select(document_version_table.c.source_hash).where(
                                and_(
                                    document_version_table.c.tenant_id == actor.tenant_id,
                                    document_version_table.c.document_id == doc_id,
                                    document_version_table.c.id == ver_id,
                                )
                            )
                        ).first()
                        src_hash = bytes(doc_ver.source_hash) if doc_ver else b"\x00" * 32

                        conn.execute(
                            eval_gold_table.insert().values(
                                tenant_id=actor.tenant_id,
                                id=str(uuid.uuid4()),
                                case_id=case_id,
                                fact_group=fact_name,
                                alternative_no=alt_no,
                                origin_system=env,
                                origin_tenant_id=actor.tenant_id,
                                origin_document_id=doc_id,
                                version_hash=src_hash,
                                anchor_json=alt,
                                relevance_grade=grade,
                                created_at=now,
                                updated_at=now,
                            )
                        )

            conn.commit()
            return {"count": len(inputs), "datasetVersionId": version_id}

    @staticmethod
    def update_case(actor: Actor, case_id: str, body: Dict[str, Any]) -> Dict[str, Any]:
        EvaluationCatalog.require_admin(actor)
        with get_db_connection() as conn:
            c = conn.execute(
                select(eval_case_table).where(
                    and_(
                        eval_case_table.c.tenant_id == actor.tenant_id,
                        eval_case_table.c.id == case_id,
                    )
                )
            ).first()
            if not c:
                raise ApiException(404, "RESOURCE_NOT_FOUND", "用例不存在")
            merged = case_view(c)
            merged.update(body)
            merged["caseKey"] = c.case_key
            version_id = str(c.dataset_version_id)

        return EvaluationCatalog.save_cases(actor, version_id, [merged])

    @staticmethod
    def delete_case(actor: Actor, case_id: str):
        EvaluationCatalog.require_admin(actor)
        with get_db_connection() as conn:
            c = conn.execute(
                select(eval_case_table.c.dataset_version_id).where(
                    and_(
                        eval_case_table.c.tenant_id == actor.tenant_id,
                        eval_case_table.c.id == case_id,
                    )
                )
            ).first()
            if not c:
                raise ApiException(404, "RESOURCE_NOT_FOUND", "用例不存在")

            EvaluationCatalog._require_draft(conn, actor, str(c.dataset_version_id))

            conn.execute(
                delete(eval_gold_table).where(
                    and_(
                        eval_gold_table.c.tenant_id == actor.tenant_id,
                        eval_gold_table.c.case_id == case_id,
                    )
                )
            )
            conn.execute(
                delete(eval_case_table).where(
                    and_(
                        eval_case_table.c.tenant_id == actor.tenant_id,
                        eval_case_table.c.id == case_id,
                    )
                )
            )
            conn.commit()

    @staticmethod
    def freeze(actor: Actor, version_id: str) -> Dict[str, Any]:
        EvaluationCatalog.require_admin(actor)
        with get_db_connection() as conn:
            EvaluationCatalog._require_draft(conn, actor, version_id)
            cs = EvaluationCatalog.cases(actor, version_id)
            if not cs:
                raise ApiException(409, "EMPTY_DATASET", "空题集不能冻结")

            group_splits: Dict[str, str] = {}
            for c in cs:
                if c.get("annotationStatus") == "DRAFT":
                    raise ApiException(409, "ANNOTATION_REQUIRED", "需要完成标注核查")

                group = str(c.get("groupKey"))
                split = str(c.get("split"))
                old_split = group_splits.setdefault(group, split)
                if old_split != split:
                    raise ApiException(409, "SPLIT_LEAKAGE", "同组追问/近重复题不能跨开发与测试分区")

                expected = obj(c.get("expected"))
                gold_groups = maps(expected.get("evidenceGroups"))
                if expected.get("status") != "ABSTAIN" and not gold_groups:
                    raise ApiException(409, "GOLD_REQUIRED", "可答题需要原文事实组")

                for g in gold_groups:
                    alts = maps(g.get("alternatives"))
                    if not alts:
                        raise ApiException(409, "GOLD_REQUIRED", "事实组缺原文")
                    for anchor in alts:
                        EvaluationCatalog._validate_anchor(conn, actor, anchor)

            m_hash = hash_payload(cs)
            now = datetime.now(timezone.utc)
            conn.execute(
                update(eval_dataset_version_table)
                .where(
                    and_(
                        eval_dataset_version_table.c.tenant_id == actor.tenant_id,
                        eval_dataset_version_table.c.id == version_id,
                    )
                )
                .values(
                    status="FROZEN",
                    manifest_hash=m_hash,
                    frozen_at=now,
                    updated_at=now,
                )
            )
            conn.commit()

            row = conn.execute(
                select(eval_dataset_version_table).where(
                    and_(
                        eval_dataset_version_table.c.tenant_id == actor.tenant_id,
                        eval_dataset_version_table.c.id == version_id,
                    )
                )
            ).first()
            return dataset_view(row)

    @staticmethod
    def _validate_anchor(conn, actor: Actor, anchor: Dict[str, Any]):
        doc_id = str(anchor.get("documentId", "")).strip()
        ver_id = str(anchor.get("versionId", "")).strip()
        quote = str(anchor.get("quote", "")).strip()
        if not doc_id or not ver_id or not quote:
            raise ApiException(400, "INVALID_INPUT", "证据锚点参数不完整")

        if not can_read_document(conn, actor, doc_id):
            raise ApiException(403, "ACCESS_DENIED", f"文档不可见: {doc_id}")

        # 在指定文档版本对应的 build/source_unit 中重新定位 Gold 引用文本；只有可复现定位的 quote 才算有效锚点。
        stmt = (
            select(source_unit_table.c.text_content)
            .select_from(
                source_unit_table.join(
                    index_build_table,
                    and_(
                        index_build_table.c.tenant_id == source_unit_table.c.tenant_id,
                        index_build_table.c.id == source_unit_table.c.build_id,
                    ),
                )
            )
            .where(
                and_(
                    index_build_table.c.tenant_id == actor.tenant_id,
                    index_build_table.c.document_id == doc_id,
                    index_build_table.c.version_id == ver_id,
                )
            )
        )
        units = conn.execute(stmt).fetchall()
        if not any(quote in (u.text_content or "") for u in units):
            raise ApiException(409, "ANCHOR_UNRESOLVED", "原文版本无法定位金标准引文")

    @staticmethod
    def create_snapshot(actor: Actor, body: Dict[str, Any]) -> Dict[str, Any]:
        EvaluationCatalog.require_admin(actor)
        name = str(body.get("name", "")).strip()
        if not name:
            raise ApiException(400, "INVALID_INPUT", "name 必填")

        raw_docs = body.get("documentIds")
        if not isinstance(raw_docs, list) or not raw_docs:
            raise ApiException(400, "INVALID_INPUT", "必须选择资料")

        doc_ids = list(dict.fromkeys(str(d) for d in raw_docs))
        approved = bool(body.get("approvedSyntheticOrRedacted", False)) or str(body.get("approval", "")) in {"SYNTHETIC", "REDACTED"}
        if not approved:
            raise ApiException(400, "SOURCE_APPROVAL_REQUIRED", "必须确认资料为模拟或已脱敏")

        now = datetime.now(timezone.utc)
        snap_id = str(uuid.uuid4())
        env = getattr(settings, "RAG_ENVIRONMENT", "demo")

        with get_db_connection() as conn:
            docs_info = []
            for doc in doc_ids:
                if not can_read_document(conn, actor, doc):
                    raise ApiException(403, "ACCESS_DENIED", f"无权访问文档: {doc}")

                d_row = conn.execute(
                    select(
                        document_table.c.id,
                        document_table.c.active_build_id,
                        index_build_table.c.version_id,
                        index_build_table.c.config_id,
                        document_version_table.c.source_hash,
                    )
                    .select_from(
                        document_table.join(
                            index_build_table,
                            and_(
                                index_build_table.c.tenant_id == document_table.c.tenant_id,
                                index_build_table.c.id == document_table.c.active_build_id,
                            ),
                        ).join(
                            document_version_table,
                            and_(
                                document_version_table.c.tenant_id == index_build_table.c.tenant_id,
                                document_version_table.c.id == index_build_table.c.version_id,
                            ),
                        )
                    )
                    .where(
                        and_(
                            document_table.c.tenant_id == actor.tenant_id,
                            document_table.c.id == doc,
                            document_table.c.status == "LIVE",
                        )
                    )
                ).first()

                if not d_row:
                    raise ApiException(404, "RESOURCE_NOT_FOUND", f"活跃文档不存在: {doc}")

                src_hex = bytes(d_row.source_hash).hex() if d_row.source_hash else ""
                docs_info.append({
                    "documentId": doc,
                    "buildId": str(d_row.active_build_id),
                    "versionId": str(d_row.version_id),
                    "sourceHash": src_hex,
                    "buildConfigId": str(d_row.config_id),
                })

            epoch_row = conn.execute(
                select(tenant_table.c.security_epoch).where(tenant_table.c.id == actor.tenant_id)
            ).first()
            epoch = epoch_row.security_epoch if epoch_row else 0

            manifest = {
                "documents": docs_info,
                "environment": env,
                "securityEpoch": epoch,
            }
            m_hash = hash_payload(manifest)

            conn.execute(
                eval_snapshot_table.insert().values(
                    tenant_id=actor.tenant_id,
                    id=snap_id,
                    name=name,
                    manifest_hash=m_hash,
                    manifest_json=manifest,
                    status="READY",
                    environment_id=env,
                    created_at=now,
                    updated_at=now,
                )
            )

            for d in docs_info:
                conn.execute(
                    eval_snapshot_document_table.insert().values(
                        tenant_id=actor.tenant_id,
                        id=str(uuid.uuid4()),
                        snapshot_id=snap_id,
                        document_id=d["documentId"],
                        version_id=d["versionId"],
                        origin_system=env,
                        origin_tenant_id=actor.tenant_id,
                        origin_document_id=d["documentId"],
                        origin_version_hash=bytes.fromhex(d["sourceHash"]) if d["sourceHash"] else b"\x00" * 32,
                        created_at=now,
                        updated_at=now,
                    )
                )
            conn.commit()

            return {"id": snap_id, "snapshotId": snap_id, "manifest": manifest}

    @staticmethod
    def snapshots(actor: Actor) -> List[Dict[str, Any]]:
        EvaluationCatalog.require_admin(actor)
        with get_db_connection() as conn:
            rows = conn.execute(
                select(eval_snapshot_table)
                .where(eval_snapshot_table.c.tenant_id == actor.tenant_id)
                .order_by(eval_snapshot_table.c.created_at.desc())
            ).fetchall()
            return [
                {
                    "id": str(r.id),
                    "name": str(r.name),
                    "status": str(r.status),
                    "environmentId": str(r.environment_id),
                    "manifest": obj(r.manifest_json),
                }
                for r in rows
            ]

    @staticmethod
    def check_snapshot(actor: Actor, snapshot_id: str, check_current: bool = False) -> List[str]:
        with get_db_connection() as conn:
            snap = conn.execute(
                select(eval_snapshot_table).where(
                    and_(
                        eval_snapshot_table.c.tenant_id == actor.tenant_id,
                        eval_snapshot_table.c.id == snapshot_id,
                    )
                )
            ).first()
            if not snap:
                raise ApiException(404, "RESOURCE_NOT_FOUND", "快照不存在")
            if snap.status not in {"READY", "ACTIVE"}:
                raise ApiException(409, "SNAPSHOT_UNAVAILABLE", "快照已失效")

            manifest = obj(snap.manifest_json)
            if check_current:
                tenant_epoch = conn.execute(
                    select(tenant_table.c.security_epoch).where(tenant_table.c.id == actor.tenant_id)
                ).scalar() or 0
                if str(manifest.get("securityEpoch")) != str(tenant_epoch):
                    raise ApiException(409, "ACL_SNAPSHOT_CHANGED", "权限已改变，请新建快照")

            doc_ids = []
            for d in maps(manifest.get("documents")):
                doc_id = str(d.get("documentId"))
                if not can_read_document(conn, actor, doc_id):
                    raise ApiException(403, "ACCESS_DENIED", f"文档不可见: {doc_id}")
                if check_current:
                    curr_doc = conn.execute(
                        select(document_table.c.active_build_id).where(
                            and_(
                                document_table.c.tenant_id == actor.tenant_id,
                                document_table.c.id == doc_id,
                            )
                        )
                    ).first()
                    if not curr_doc or str(curr_doc.active_build_id) != str(d.get("buildId")):
                        raise ApiException(409, "SOURCE_CHANGED", "生效构建已改变，请新建快照")
                doc_ids.append(doc_id)
            return doc_ids

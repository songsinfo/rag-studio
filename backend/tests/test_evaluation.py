import json
import os
import pytest
from datetime import datetime, timezone
from sqlalchemy import select, and_, delete

from app.config import settings
from app.db import get_db_connection
from app.errors import ApiException
from app.identity.deps import Actor
from app.evaluation.metrics import EvaluationMetrics
from app.evaluation.catalog import EvaluationCatalog, hash_payload, obj, maps
from app.evaluation.runner import EvaluationRunner, check_no_model, number
from app.models import (
    tenant_table,
    app_user_table,
    knowledge_base_table,
    document_table,
    document_version_table,
    index_build_table,
    source_unit_table,
    chunk_table,
    eval_dataset_version_table,
    eval_case_table,
    eval_gold_table,
    eval_snapshot_table,
    eval_snapshot_document_table,
    eval_run_table,
    eval_result_table,
    eval_review_table,
    config_revision_table,
)


# ==============================================================================
# 第 1 组：算法指标测试，并与 Java 基线公式保持一致。
# ==============================================================================

def test_evaluation_rag_is_executed_by_worker_outside_test_mode(monkeypatch, eval_env):
    from types import SimpleNamespace
    import app.qa.service as qa_service
    import app.evaluation.runner as evaluation_runner
    from app.models import job_table, qa_run_table
    from app.qa.service import QaService
    from app.runtime.worker import DurableJobWorker

    # 走真实 submit/poll/Worker 链路，避免测试环境的同步执行掩盖缺失的调度。
    monkeypatch.setattr(qa_service, "is_test_env", lambda: False)
    worker = DurableJobWorker(kinds={"QA_RUN"})
    executed = []

    def execute_pending_run(_):
        QaService.recover_unfinished_runs()
        assert worker.run_once(), "评估 RAG 必须有可领取的 QA_RUN Job"
        executed.append(True)

    monkeypatch.setattr(evaluation_runner, "time", SimpleNamespace(
        monotonic=lambda: 0.0, sleep=execute_pending_run,
    ))
    actor = eval_env["admin"]
    case = {"caseKey": "worker-rag", "question": {"text": "你好"}}
    manifest = {"mode": "RAG", "configId": "cfg-enterprise-retrieval-v2"}
    snapshot = EvaluationCatalog.create_snapshot(actor, {
        "name": "Worker RAG", "documentIds": [eval_env["doc_id"]],
        "approvedSyntheticOrRedacted": True,
    })
    scope = {**snapshot["manifest"], "snapshotId": snapshot["id"]}
    detail, run_ids = EvaluationRunner._run_qa_turns(
        actor, actor, "review-eval", "review-result", case, manifest, scope,
    )
    assert detail["status"] == "COMPLETED"
    assert detail["answer"]
    reused_detail, reused_ids = EvaluationRunner._run_qa_turns(
        actor, actor, "review-eval", "review-result", case, manifest, scope,
    )
    assert reused_ids == run_ids
    assert reused_detail["status"] == "COMPLETED"
    assert len(executed) == 1
    with get_db_connection() as conn:
        run = conn.execute(select(qa_run_table).where(qa_run_table.c.id == run_ids[0])).first()
        jobs = conn.execute(select(job_table).where(job_table.c.target_id == run_ids[0])).all()
    assert run.started_at is not None
    assert len(jobs) == 1
    assert jobs[0].status == "COMPLETED"


def test_one_of_three_facts_is_a_hit_but_only_third_recall():
    m = EvaluationMetrics.retrieval(
        groups=[{"a"}, {"b"}, {"c"}],
        ranked=[{"a"}],
        k=5,
    )
    assert m["hit"] == 1.0
    assert abs(m["recall"] - 1.0 / 3) < 1e-6
    assert m["mrr"] == 1.0


def test_duplicate_chunks_and_alternative_evidence_do_not_invent_coverage():
    m = EvaluationMetrics.retrieval(
        groups=[{"a", "a2"}, {"b"}],
        ranked=[{"a2"}, {"a"}, {"a2"}],
        k=5,
    )
    assert m["recall"] == 0.5


def test_empty_gold_is_not_perfect_and_failed_answerable_question_is_zero():
    assert EvaluationMetrics.retrieval(groups=[], ranked=[], k=5)["recall"] is None
    assert EvaluationMetrics.retrieval(groups=[{"a"}], ranked=[], k=5)["recall"] == 0.0


def test_zero_claims_cannot_score_perfect_faithfulness():
    assert EvaluationMetrics.ratio(0, 0) is None
    assert EvaluationMetrics.ratio(5, 10) == 0.5


def test_dcg_uses_graded_relevance_and_rejects_unknown_grades():
    assert abs(EvaluationMetrics.ndcg([3, 2, 0], [3, 2, 0], 3) - 1.0) < 1e-9
    assert EvaluationMetrics.ndcg([0, 2, 3], [3, 2, 0], 3) < 1.0
    assert EvaluationMetrics.ndcg([0], [0], 1) is None

    with pytest.raises(ValueError, match="Grade must be 0..3"):
        EvaluationMetrics.ndcg([-1], [3], 1)

    with pytest.raises(ValueError, match="Grade must be 0..3"):
        EvaluationMetrics.ndcg([4], [3], 1)


def test_retrieval_result_includes_ndcg_for_the_same_ranked_candidates():
    m = EvaluationMetrics.retrieval(
        groups=[{"a"}, {"b"}],
        ranked=[{"b"}, {"a"}],
        k=2,
        ranked_grades=[1, 3],
        ideal_grades=[3, 1],
    )
    assert m["recall"] == 1.0
    assert m["ndcg"] is not None
    assert 0.0 < m["ndcg"] < 1.0


def test_nearest_rank_percentile_uses_all_observed_durations():
    assert EvaluationMetrics.percentile([1, 2, 3, 1000], 0.95) == 1000
    assert EvaluationMetrics.percentile([100, 200, 300, 400], 0.5) == 200
    assert EvaluationMetrics.percentile([], 0.5) is None


def test_retrieval_only_permits_bm25_without_model_transforms_or_reranking():
    valid_bm25 = {
        "retrieval": {"mode": "BM25"},
        "query": {"rewrite": False, "hyde": False, "multiQuery": False},
        "post": {"rerank": False, "mmr": False},
    }
    check_no_model(valid_bm25)

    with pytest.raises(ApiException) as exc1:
        check_no_model({"retrieval": {"mode": "DENSE"}})
    assert exc1.value.code == "MODEL_REQUIRED"

    # mode 缺失时必须与 RetrievalConfig.from_payload 使用相同默认逻辑：
    # 默认归一化为 HYBRID；该模式会调用 query embedding，因此不能被误判为 zero-model。
    with pytest.raises(ApiException) as exc_default:
        check_no_model({"retrieval": {"topK": 5}})
    assert exc_default.value.code == "MODEL_REQUIRED"

    with pytest.raises(ApiException) as exc2:
        check_no_model({"retrieval": {"mode": "BM25"}, "query": {"rewrite": True}})
    assert exc2.value.code == "MODEL_REQUIRED"

    with pytest.raises(ApiException) as exc3:
        check_no_model({"retrieval": {"mode": "BM25"}, "post": {"rerank": True}})
    assert exc3.value.code == "MODEL_REQUIRED"


def test_number_validation():
    assert number("3.14") == 3.14
    assert number(10) == 10.0
    with pytest.raises(ApiException):
        number("abc")
    with pytest.raises(ApiException):
        number("NaN")
    with pytest.raises(ApiException):
        number("Infinity")


def test_deterministic_hash_matches_identical_content():
    m1 = {"caseKey": "K1", "split": "DEV"}
    m2 = {"caseKey": "K1", "split": "DEV"}
    assert hash_payload(m1) == hash_payload(m2)


def test_object_and_map_parsers_validate_structure():
    assert obj(None) == {}
    assert maps(None) == []
    with pytest.raises(ApiException):
        maps("not-a-list")


# ==============================================================================
# 第 2 组：题集生命周期、split 泄漏与 Gold anchor 校验测试。
# ==============================================================================

@pytest.fixture
def eval_env():
    """Setup isolated test tenant, admin actor, and dummy seeded documents."""
    t_id = "eval-test-tenant"
    admin_id = "eval-admin-user"
    now = datetime.now(timezone.utc)

    with get_db_connection() as conn:
        # 先清理同名测试数据，保证重复运行时结果不受旧记录影响。
        conn.execute(delete(eval_review_table).where(eval_review_table.c.tenant_id == t_id))
        conn.execute(delete(eval_result_table).where(eval_result_table.c.tenant_id == t_id))
        conn.execute(delete(eval_run_table).where(eval_run_table.c.tenant_id == t_id))
        conn.execute(delete(eval_snapshot_document_table).where(eval_snapshot_document_table.c.tenant_id == t_id))
        conn.execute(delete(eval_snapshot_table).where(eval_snapshot_table.c.tenant_id == t_id))
        conn.execute(delete(eval_gold_table).where(eval_gold_table.c.tenant_id == t_id))
        conn.execute(delete(eval_case_table).where(eval_case_table.c.tenant_id == t_id))
        conn.execute(delete(eval_dataset_version_table).where(eval_dataset_version_table.c.tenant_id == t_id))
        conn.execute(delete(config_revision_table).where(config_revision_table.c.tenant_id == t_id))
        conn.execute(delete(chunk_table).where(chunk_table.c.tenant_id == t_id))
        conn.execute(delete(source_unit_table).where(source_unit_table.c.tenant_id == t_id))
        conn.execute(delete(index_build_table).where(index_build_table.c.tenant_id == t_id))
        conn.execute(delete(document_version_table).where(document_version_table.c.tenant_id == t_id))
        conn.execute(delete(document_table).where(document_table.c.tenant_id == t_id))
        from app.models import kb_grant_table
        conn.execute(delete(kb_grant_table).where(kb_grant_table.c.tenant_id == t_id))
        conn.execute(delete(knowledge_base_table).where(knowledge_base_table.c.tenant_id == t_id))
        conn.execute(delete(app_user_table).where(app_user_table.c.tenant_id == t_id))
        conn.execute(delete(tenant_table).where(tenant_table.c.id == t_id))

        # 写入测试租户，作为后续用户、知识库和评估数据的隔离边界。
        conn.execute(
            tenant_table.insert().values(
                id=t_id,
                code="eval_test",
                name="Eval Test Tenant",
                security_epoch=1,
                active_retrieval_config_id=None,
                created_at=now,
            )
        )
        # 写入管理员用户，用于执行评估配置与题集管理操作。
        conn.execute(
            app_user_table.insert().values(
                tenant_id=t_id,
                id=admin_id,
                login_name="eval_admin",
                password_hash="mock",
                is_admin=True,
                status="ACTIVE",
                created_at=now,
                updated_at=now,
            )
        )
        # 写入普通 operator 用户，验证非管理员权限路径。
        conn.execute(
            app_user_table.insert().values(
                tenant_id=t_id,
                id="eval_operator_id",
                login_name="operator",
                password_hash="mock",
                is_admin=False,
                status="ACTIVE",
                created_at=now,
                updated_at=now,
            )
        )
        # 写入测试知识库，作为 Gold 文档与检索快照的父资源。
        kb_id = "eval-kb-01"
        conn.execute(
            knowledge_base_table.insert().values(
                tenant_id=t_id,
                id=kb_id,
                name="Eval Knowledge Base",
                status="LIVE",
                created_at=now,
                updated_at=now,
            )
        )
        conn.execute(
            kb_grant_table.insert().values(
                tenant_id=t_id,
                id="grant-kb-01",
                kb_id=kb_id,
                principal_type="ALL",
                user_id=None,
                group_id=None,
                created_at=now,
                updated_at=now,
            )
        )
        # 写入文档和对应 build，构造可被快照/anchor 引用的真实语料版本。
        doc_id = "doc-eval-01"
        ver_id = "ver-eval-01"
        build_id = "build-eval-01"
        conn.execute(
            document_table.insert().values(
                tenant_id=t_id,
                id=doc_id,
                kb_id=kb_id,
                title="Evaluation Spec Document",
                acl_mode="INHERIT",
                status="LIVE",
                active_build_id=build_id,
                created_at=now,
                updated_at=now,
            )
        )
        conn.execute(
            document_version_table.insert().values(
                tenant_id=t_id,
                id=ver_id,
                document_id=doc_id,
                version_no=1,
                source_artifact_id=None,
                source_hash=b"mock-source-hash-32-bytes-long!!",
                original_name="spec.md",
                media_type="text/markdown",
                created_by=admin_id,
                created_at=now,
                updated_at=now,
            )
        )
        conn.execute(
            index_build_table.insert().values(
                tenant_id=t_id,
                id=build_id,
                document_id=doc_id,
                version_id=ver_id,
                build_no=1,
                config_id="cfg-default",
                status="READY",
                embedding_dims=1024,
                capabilities={"bm25": True, "dense": True},
                manifest_hash=b"0" * 32,
                fence=0,
                created_at=now,
                updated_at=now,
            )
        )
        # 写入包含目标 quote 的 source_unit，用于验证 Gold 文本能够重新定位。
        conn.execute(
            source_unit_table.insert().values(
                tenant_id=t_id,
                id="unit-eval-01",
                build_id=build_id,
                document_id=doc_id,
                unit_key="u1",
                kind="PARAGRAPH",
                text_content="知识库系统采用结构化分块策略保留Markdown标题层级，提供端到端严谨召回保证。",
                location={},
                machine_generated=False,
                content_hash=b"0" * 32,
                created_at=now,
                updated_at=now,
            )
        )
        # 写入包含同一 quote 的 chunk，覆盖 chunk 级命中与 provenance 校验。
        chunk_id = "chunk-eval-01"
        chunk_text = "知识库系统采用结构化分块策略保留Markdown标题层级，提供端到端严谨召回保证。"
        conn.execute(
            chunk_table.insert().values(
                tenant_id=t_id,
                id=chunk_id,
                build_id=build_id,
                parent_id=None,
                kind="PARAGRAPH",
                ordinal=1,
                text_content=chunk_text,
                token_count=50,
                content_hash=b"0" * 32,
                created_at=now,
                updated_at=now,
            )
        )
        conn.commit()

    from app.retrieval.milvus_service import MilvusHybridService
    MilvusHybridService.upsert_chunks(
        tenant_id=t_id,
        build_id=build_id,
        chunks=[{"id": chunk_id, "text_content": chunk_text}],
        kb_id=kb_id,
        doc_id=doc_id,
        version_id=ver_id,
    )

    admin = Actor(tenant_id=t_id, user_id=admin_id, admin=True)
    return {
        "admin": admin,
        "doc_id": doc_id,
        "ver_id": ver_id,
        "quote": "结构化分块策略保留Markdown标题层级",
    }


def test_dataset_lifecycle_and_freeze_validation(eval_env):
    admin = eval_env["admin"]
    doc_id = eval_env["doc_id"]
    ver_id = eval_env["ver_id"]
    quote = eval_env["quote"]

    # 第 1 步：创建评估题集及初始版本。
    ds = EvaluationCatalog.create_dataset(admin, {"name": "Smoke Dataset"})
    assert ds["name"] == "Smoke Dataset"
    assert ds["revision"] == 1
    assert ds["status"] == "DRAFT"
    ds_id = ds["id"]

    # 第 2 步：批量导入 case，建立问题、预期与 Gold 结构。
    cases_payload = [
        {
            "caseKey": "case-01",
            "groupKey": "grp-01",
            "split": "DEV",
            "category": "FACT",
            "question": {"text": "系统分块策略是什么？"},
            "persona": {"loginName": "operator"},
            "expected": {
                "status": "ANSWER",
                "evidenceGroups": [
                    {
                        "fact": "分块策略",
                        "alternatives": [
                            {
                                "documentId": doc_id,
                                "versionId": ver_id,
                                "quote": quote,
                                "relevanceGrade": 3,
                            }
                        ],
                    }
                ],
            },
            "annotationStatus": "DRAFT",
        }
    ]
    res = EvaluationCatalog.save_cases(admin, ds_id, cases_payload)
    assert res["count"] == 1

    # 第 3 步：annotationStatus 仍为 DRAFT 时尝试 freeze，必须返回 ANNOTATION_REQUIRED。
    with pytest.raises(ApiException) as exc:
        EvaluationCatalog.freeze(admin, ds_id)
    assert exc.value.code == "ANNOTATION_REQUIRED"

    # 第 4 步：导入存在 split 泄漏的 case，必须返回 SPLIT_LEAKAGE。
    leak_case = {
        "caseKey": "case-02-leak",
        "groupKey": "grp-01",  # Same groupKey as case-01
        "split": "TEST",       # Different split!
        "category": "FACT",
        "question": {"text": "系统分块策略在盲测里的表现？"},
        "persona": {"loginName": "operator"},
        "expected": {
            "status": "ANSWER",
            "evidenceGroups": [
                {
                    "fact": "分块策略",
                    "alternatives": [
                        {
                            "documentId": doc_id,
                            "versionId": ver_id,
                            "quote": quote,
                            "relevanceGrade": 3,
                        }
                    ],
                }
            ],
        },
        "annotationStatus": "AUTO_REVIEWED",
    }
    # 先把 case-01 更新为 AUTO_REVIEWED，排除 annotation 状态对后续 split 校验的干扰。
    EvaluationCatalog.update_case(admin, EvaluationCatalog.cases(admin, ds_id)[0]["id"], {"annotationStatus": "AUTO_REVIEWED"})
    # 保存专门构造的泄漏 case，用于 freeze 阶段检测跨 split 重复。
    EvaluationCatalog.save_cases(admin, ds_id, [leak_case])

    # 此时 freeze 应准确检测 SPLIT_LEAKAGE，而不是被其他前置校验抢先失败。
    with pytest.raises(ApiException) as exc_leak:
        EvaluationCatalog.freeze(admin, ds_id)
    assert exc_leak.value.code == "SPLIT_LEAKAGE"

    # 删除泄漏 case，继续验证后续 anchor 和冻结流程。
    leak_case_id = [c["id"] for c in EvaluationCatalog.cases(admin, ds_id) if c["caseKey"] == "case-02-leak"][0]
    EvaluationCatalog.delete_case(admin, leak_case_id)

    # 第 5 步：使用无法在语料中定位的 quote 冻结，必须返回 ANCHOR_UNRESOLVED。
    invalid_quote_case = {
        "caseKey": "case-03-bad-quote",
        "groupKey": "grp-03",
        "split": "DEV",
        "category": "FACT",
        "question": {"text": "不存在的引用？"},
        "persona": {"loginName": "operator"},
        "expected": {
            "status": "ANSWER",
            "evidenceGroups": [
                {
                    "fact": "假事实",
                    "alternatives": [
                        {
                            "documentId": doc_id,
                            "versionId": ver_id,
                            "quote": "这是一段在原文中完全不存在的虚构引文内容",
                            "relevanceGrade": 3,
                        }
                    ],
                }
            ],
        },
        "annotationStatus": "AUTO_REVIEWED",
    }
    EvaluationCatalog.save_cases(admin, ds_id, [invalid_quote_case])
    with pytest.raises(ApiException) as exc_anchor:
        EvaluationCatalog.freeze(admin, ds_id)
    assert exc_anchor.value.code == "ANCHOR_UNRESOLVED"

    # 删除非法 quote case，恢复到可正常冻结的数据集。
    bad_id = [c["id"] for c in EvaluationCatalog.cases(admin, ds_id) if c["caseKey"] == "case-03-bad-quote"][0]
    EvaluationCatalog.delete_case(admin, bad_id)

    # 第 6 步：所有校验通过后成功冻结题集版本。
    frozen_ds = EvaluationCatalog.freeze(admin, ds_id)
    assert frozen_ds["status"] == "FROZEN"
    assert frozen_ds["frozenAt"] is not None

    # 第 7 步：尝试修改已冻结题集，必须返回 DATASET_FROZEN，保证实验输入不可变。
    with pytest.raises(ApiException) as exc_mod:
        EvaluationCatalog.save_cases(admin, ds_id, cases_payload)
    assert exc_mod.value.code == "DATASET_FROZEN"

    # 第 8 步：复制冻结版本时创建新的 DRAFT 版本，后续修改必须发生在新版本上。
    copied = EvaluationCatalog.copy_version(admin, ds_id)
    assert copied["revision"] == 2
    assert copied["status"] == "DRAFT"
    copied_cases = EvaluationCatalog.cases(admin, copied["id"])
    assert len(copied_cases) == 1
    assert copied_cases[0]["caseKey"] == "case-01"


# ==============================================================================
# 第 3 组：语料快照、检索配置与 EvaluationRunner 集成测试。
# ==============================================================================

def test_snapshot_config_and_evaluation_run(eval_env):
    from fastapi.testclient import TestClient
    from app.main import app
    from app.identity.security import issue_csrf_token

    client = TestClient(app)
    admin = eval_env["admin"]
    doc_id = eval_env["doc_id"]
    ver_id = eval_env["ver_id"]
    quote = eval_env["quote"]

    # 生成测试登录会话和 CSRF token，后续管理接口全部走真实鉴权链路。
    session_token = "test-eval-session-token"
    with get_db_connection() as conn:
        from app.models import auth_session_table
        from app.identity.security import hash_token
        import uuid
        now = datetime.now(timezone.utc)
        conn.execute(
            auth_session_table.insert().values(
                tenant_id=admin.tenant_id,
                id=str(uuid.uuid4()),
                user_id=admin.user_id,
                token_hash=hash_token(session_token),
                expires_at=datetime(2030, 1, 1, tzinfo=timezone.utc),
                revoked_at=None,
                created_at=now,
                updated_at=now,
            )
        )
        conn.commit()

    csrf = issue_csrf_token(session_token, settings.RAG_CSRF_SECRET)
    headers = {"x-csrf-token": csrf}
    cookies = {"RAG_SESSION": session_token, "RAG_CSRF": csrf}

    # 第 1 步：通过管理接口创建并发布 RETRIEVAL 配置，验证评估引用的是已发布不可变配置。
    cfg_resp = client.post(
        "/api/v1/admin/configs",
        json={
            "kind": "RETRIEVAL",
            "name": "eval_bm25_cfg",
            "payload": {
                "retrieval": {"mode": "BM25"},
                "query": {"rewrite": False, "hyde": False, "multiQuery": False},
                "post": {"rerank": False, "mmr": False},
            },
        },
        headers=headers,
        cookies=cookies,
    )
    assert cfg_resp.status_code == 201
    cfg_id = cfg_resp.json()["id"]

    pub_resp = client.post(
        f"/api/v1/admin/configs/{cfg_id}/publish",
        headers=headers,
        cookies=cookies,
    )
    assert pub_resp.status_code == 200
    assert pub_resp.json()["status"] == "PUBLISHED"

    # 读取配置列表，确认刚发布的版本可被后续 run 选择。
    list_cfg = client.get("/api/v1/admin/configs", cookies=cookies)
    assert list_cfg.status_code == 200
    assert any(c["id"] == cfg_id for c in list_cfg.json())

    # 第 2 步：创建语料快照，冻结当前 active build 与安全版本。
    snap_resp = client.post(
        "/api/v1/admin/eval/snapshots",
        json={
            "name": "Eval Baseline Snapshot",
            "documentIds": [doc_id],
            "approvedSyntheticOrRedacted": True,
        },
        headers=headers,
        cookies=cookies,
    )
    assert snap_resp.status_code == 201
    snap_id = snap_resp.json()["id"]

    # 第 3 步：创建并冻结题集，保证 run 启动后 Gold 不再变化。
    ds = EvaluationCatalog.create_dataset(admin, {"name": "Eval Test Suite"})
    ds_id = ds["id"]
    EvaluationCatalog.save_cases(
        admin,
        ds_id,
        [
            {
                "caseKey": "smoke-qa-01",
                "groupKey": "grp-smoke-01",
                "split": "TEST",
                "category": "FACT",
                "question": {"text": "知识库系统采用什么分块策略？"},
                "persona": {"loginName": "operator"},
                "expected": {
                    "status": "ANSWER",
                    "evidenceGroups": [
                        {
                            "fact": "分块策略",
                            "alternatives": [
                                {
                                    "documentId": doc_id,
                                    "versionId": ver_id,
                                    "quote": quote,
                                    "relevanceGrade": 3,
                                }
                            ],
                        }
                    ],
                },
                "annotationStatus": "AUTO_REVIEWED",
            },
            {
                "caseKey": "smoke-qa-02",
                "groupKey": "grp-smoke-02",
                "split": "TEST",
                "category": "FACT",
                "question": {"text": "不可回答的问题？"},
                "persona": {"loginName": "operator"},
                "expected": {
                    "status": "ABSTAIN",
                    "evidenceGroups": [],
                },
                "annotationStatus": "AUTO_REVIEWED",
            },
        ],
    )
    EvaluationCatalog.freeze(admin, ds_id)

    # 第 4 步：先估算评估费用，验证预算门槛与模型调用上限。
    est_resp = client.post(
        "/api/v1/admin/eval/estimates",
        json={
            "datasetVersionId": ds_id,
            "snapshotId": snap_id,
            "configId": cfg_id,
            "split": "TEST",
            "repeats": 1,
            "mode": "RETRIEVAL_ONLY",
        },
        headers=headers,
        cookies=cookies,
    )
    assert est_resp.status_code == 200
    est = est_resp.json()
    assert est["plannedQuestions"] == 2
    assert est["costStatus"] == "NO_MODEL_CALLS"
    assert est["estimateId"] is not None

    # 第 5 步：使用 estimateId 启动真实评估 run。
    start_resp = client.post(
        "/api/v1/admin/eval/runs",
        json={
            "datasetVersionId": ds_id,
            "snapshotId": snap_id,
            "configId": cfg_id,
            "split": "TEST",
            "repeats": 1,
            "mode": "RETRIEVAL_ONLY",
            "estimateId": est["estimateId"],
            "maxCostCny": 5,
        },
        headers={"x-csrf-token": csrf, "Idempotency-Key": "eval-key-001"},
        cookies=cookies,
    )
    assert start_resp.status_code == 201
    run_id = start_resp.json()["runId"]

    # 持续轮询直到评估进入终态，避免仅验证任务创建。
    import time
    for _ in range(60):
        run_detail = client.get(f"/api/v1/admin/eval/runs/{run_id}", cookies=cookies).json()
        if run_detail.get("status") in {"COMPLETED", "FAILED"}:
            break
        time.sleep(0.5)

    assert run_detail["status"] == "COMPLETED"
    metrics = run_detail["metrics"]
    assert metrics["planned"] == 2
    assert metrics["completed"] == 2
    assert metrics["failed"] == 0

    # 第 6 步：逐题检查结果状态、检索指标与语义判定。
    results_resp = client.get(f"/api/v1/admin/eval/runs/{run_id}/results", cookies=cookies)
    assert results_resp.status_code == 200
    cases_res = results_resp.json()
    assert len(cases_res) == 2
    res_keys = {c["caseKey"] for c in cases_res}
    assert "smoke-qa-01" in res_keys
    assert "smoke-qa-02" in res_keys
    answerable = next(c for c in cases_res if c['caseKey'] == 'smoke-qa-01')
    assert answerable['judge']['evaluationSample']['contractVersion'] == 'eobs-v1'
    assert answerable['judge']['evaluationSample']['actualContext']['status'] == 'UNKNOWN'
    stages = answerable['metrics']['stageCoverage']
    assert any(m['stage'] == 'bm25_recall' and m['status'] == 'OK'
               and m['value'] == answerable['metrics']['recall'] for m in stages)
    assert any(m['stage'] == 'finalize_hits' and m['status'] == 'OK' for m in stages)
    assert metrics['metricCoverage']

    # 第 7 步：为结果添加人工 review，验证人工标签覆盖/补充自动 Judge 的机制。
    res_id = cases_res[0]["id"]
    rev_resp = client.post(
        f"/api/v1/admin/eval/results/{res_id}/reviews",
        json={"labels": {"correctness": "CORRECT"}, "reason": "人工核验通过"},
        headers=headers,
        cookies=cookies,
    )
    assert rev_resp.status_code == 201

    # 重新读取结果，确认人工 review 已正确附着并进入 effective judgment。
    recheck_res = client.get(f"/api/v1/admin/eval/runs/{run_id}/results", cookies=cookies).json()
    r0 = [r for r in recheck_res if r["id"] == res_id][0]
    assert len(r0["reviews"]) == 1
    assert r0["reviews"][0]["labels"]["correctness"] == "CORRECT"

    # 第 8 步：测试结果导出，确保离线分析可复现当前 run。
    exp_resp = client.post(
        f"/api/v1/admin/eval/runs/{run_id}/exports",
        json={"mode": "REDACTED"},
        headers=headers,
        cookies=cookies,
    )
    assert exp_resp.status_code == 200
    assert exp_resp.json()["mode"] == "REDACTED"
    assert len(exp_resp.json()["cases"]) == 2

    # 第 9 步：测试两个 manifest 可比的 run 之间的指标比较。
    # 使用相同参数启动第二轮 run，构造合法的同条件对照组。
    start_resp2 = client.post(
        "/api/v1/admin/eval/runs",
        json={
            "datasetVersionId": ds_id,
            "snapshotId": snap_id,
            "configId": cfg_id,
            "split": "TEST",
            "repeats": 1,
            "mode": "RETRIEVAL_ONLY",
            "estimateId": est["estimateId"],
            "maxCostCny": 5,
        },
        headers={"x-csrf-token": csrf, "Idempotency-Key": "eval-key-002"},
        cookies=cookies,
    )
    run_id2 = start_resp2.json()["runId"]
    for _ in range(20):
        r2 = client.get(f"/api/v1/admin/eval/runs/{run_id2}", cookies=cookies).json()
        if r2.get("status") in {"COMPLETED", "FAILED"}:
            break
        time.sleep(0.5)

    comp_resp = client.get(f"/api/v1/admin/eval/comparisons?runIds={run_id},{run_id2}", cookies=cookies)
    assert comp_resp.status_code == 200
    comp = comp_resp.json()
    assert comp["comparable"] is True
    assert len(comp["runs"]) == 2

    # 第 10 步：与 artifacts/baseline-report.json 中合成格式示例基线做差异比较。
    diff_resp = client.get(f"/api/v1/admin/eval/runs/{run_id}/baseline-diff", cookies=cookies)
    assert diff_resp.status_code == 200
    diff = diff_resp.json()
    assert diff["runId"] == run_id
    assert diff["baselineBackend"] == "synthetic-example"
    assert "metricDeltas" in diff
    assert "configurationDifferences" in diff
    assert "recallAt5" in diff["metricDeltas"]


def test_eval_smoke_30_dataset_conformance():
    """Verify datasets/eval_smoke_30.jsonl conformance: 30 cases, Chinese text, codes, and paths."""
    cur = os.path.abspath(__file__)
    smoke_file = ""
    for _ in range(6):
        cur = os.path.dirname(cur)
        candidate = os.path.join(cur, "datasets", "eval_smoke_30.jsonl")
        if os.path.exists(candidate):
            smoke_file = candidate
            break

    assert os.path.exists(smoke_file), f"Smoke dataset not found at {smoke_file}"

    with open(smoke_file, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]

    assert len(lines) == 30

    case_keys = set()
    group_splits = {}
    has_chinese = False
    has_code_or_path = False

    for idx, line in enumerate(lines, start=1):
        c = json.loads(line)
        for req in ["caseKey", "groupKey", "split", "category", "question", "persona", "expected", "annotationStatus"]:
            assert req in c, f"Line {idx} missing {req}"

        k = c["caseKey"]
        assert k not in case_keys
        case_keys.add(k)

        split = c["split"].upper()
        assert split in {"DEV", "TEST"}

        grp = c["groupKey"]
        old_split = group_splits.setdefault(grp, split)
        assert old_split == split, f"Group {grp} leaks across splits"

        full_text = json.dumps(c, ensure_ascii=False)
        if any("\u4e00" <= ch <= "\u9fff" for ch in full_text):
            has_chinese = True
        if any(term in full_text for term in ["CSRF", "API", "H1-H6", "TRASH", "1024", "doc-sys"]):
            has_code_or_path = True

    assert has_chinese
    assert has_code_or_path


def test_single_case_failure_retention(eval_env):
    """Verify that single case failure is retained as FAILED with errorCode and never disguised."""
    admin = eval_env["admin"]
    doc_id = eval_env["doc_id"]
    ver_id = eval_env["ver_id"]
    quote = eval_env["quote"]

    # 第 1 步：创建冻结题集与语料快照，构造失败场景的稳定输入。
    ds = EvaluationCatalog.create_dataset(admin, {"name": "Failure Retention Test Suite"})
    ds_id = ds["id"]
    EvaluationCatalog.save_cases(
        admin,
        ds_id,
        [
            {
                "caseKey": "good-case",
                "groupKey": "grp-f-01",
                "split": "TEST",
                "category": "FACT",
                "question": {"text": "结构化分块策略"},
                "persona": {"loginName": "operator"},
                "expected": {
                    "status": "ANSWER",
                    "evidenceGroups": [
                        {
                            "fact": "正常事实",
                            "alternatives": [
                                {
                                    "documentId": doc_id,
                                    "versionId": ver_id,
                                    "quote": quote,
                                    "relevanceGrade": 3,
                                }
                            ],
                        }
                    ],
                },
                "annotationStatus": "AUTO_REVIEWED",
            },
            {
                "caseKey": "failing-case",
                "groupKey": "grp-f-02",
                "split": "TEST",
                "category": "FACT",
                "question": {"text": "将会失败的题目？"},
                "persona": {"loginName": "operator"},
                "expected": {
                    "status": "ANSWER",
                    "evidenceGroups": [
                        {
                            "fact": "失败事实",
                            "alternatives": [
                                {
                                    "documentId": doc_id,
                                    "versionId": ver_id,
                                    "quote": quote,
                                    "relevanceGrade": 3,
                                }
                            ],
                        }
                    ],
                },
                "annotationStatus": "AUTO_REVIEWED",
            },
        ],
    )
    EvaluationCatalog.freeze(admin, ds_id)
    snap = EvaluationCatalog.create_snapshot(admin, {"name": "Fail Test Snap", "documentIds": [doc_id], "approvedSyntheticOrRedacted": True})
    snap_id = snap["id"]

    # 注入 RETRIEVAL mock 配置，聚焦验证 runner 状态处理而不是外部检索服务。
    with get_db_connection() as conn:
        cfg_id = "cfg-fail-retention-test"
        now = datetime.now(timezone.utc)
        conn.execute(
            config_revision_table.insert().values(
                tenant_id=admin.tenant_id,
                id=cfg_id,
                kind="RETRIEVAL",
                name="fail_test_cfg",
                revision=1,
                sha256=b"0" * 32,
                payload={
                    "retrieval": {"mode": "BM25"},
                    "query": {"rewrite": False, "hyde": False, "multiQuery": False},
                    "post": {"rerank": False, "mmr": False},
                },
                status="PUBLISHED",
                created_at=now,
                updated_at=now,
            )
        )
        conn.commit()

    est = EvaluationRunner.estimate(admin, {
        "datasetVersionId": ds_id,
        "snapshotId": snap_id,
        "configId": cfg_id,
        "split": "TEST",
        "mode": "RETRIEVAL_ONLY",
    })

    # 仅对 failing-case 模拟 Milvus 搜索失败，验证单题失败不会让其他结果从计划分母消失。
    from unittest.mock import patch
    from app.retrieval.milvus_service import MilvusHybridService

    orig_search = MilvusHybridService.search

    def mock_search(*args, **kwargs):
        query = kwargs.get("query", "")
        if "将会失败" in query:
            raise RuntimeError("SIMULATED_INDEX_TIMEOUT")
        return orig_search(*args, **kwargs)

    with patch.object(MilvusHybridService, "search", side_effect=mock_search):
        run_info = EvaluationRunner.start(admin, {
            "datasetVersionId": ds_id,
            "snapshotId": snap_id,
            "configId": cfg_id,
            "split": "TEST",
            "mode": "RETRIEVAL_ONLY",
            "estimateId": est["estimateId"],
        }, "fail-retention-key-01")

        run_id = run_info["runId"]

        import time
        for _ in range(20):
            r = EvaluationRunner.get_run(admin, run_id)
            if r.get("status") in {"COMPLETED", "FAILED"}:
                break
            time.sleep(0.3)

    # 检查所有预创建结果行都进入预期终态，包括成功与失败 case。
    res_list = EvaluationRunner.results(admin, run_id)
    assert len(res_list) == 2

    good_res = [r for r in res_list if r["caseKey"] == "good-case"][0]
    bad_res = [r for r in res_list if r["caseKey"] == "failing-case"][0]

    assert good_res["status"] == "COMPLETED"
    assert good_res["errorCode"] is None

    # 关键断言：bad_res 必须保留为 FAILED 且带 errorCode，不能因为失败而被聚合逻辑过滤掉。
    assert bad_res["status"] == "FAILED"
    assert bad_res["errorCode"] == "RETRIEVAL_FAILED"
    assert "error" in bad_res["judge"]
    assert "SIMULATED_INDEX_TIMEOUT" in bad_res["judge"]["error"]

    # 检查 run 聚合统计，确认 FAILED/PENDING 等执行状态都被计入计划分母。
    run_detail = EvaluationRunner.get_run(admin, run_id)
    m = run_detail["metrics"]
    assert m["planned"] == 2
    assert m["completed"] == 1
    assert m["failed"] == 1
    assert m["taskSuccessRate"] == 1.0 / 2


# RAG Studio Backend

FastAPI 模块化后端：MySQL 保存权限与权威元数据，Milvus 保存可重建的检索索引，LiteLLM 持有使用者配置的供应商凭据，Docling 解析文档。
跨平台推荐从仓库根目录运行 `python scripts/quickstart.py init`，配置 `.runtime/demo.env` 后执行 `python scripts/quickstart.py up`；详细操作见[Quick Start](../README.md#quick-startwindows--linux--macos)。本机原生开发统一使用 `python scripts/dev.py init|deps|backend`。

本机原生开发模式可启动可选文档解析服务（统一 Docker Quick Start 已自动启动）：

```sh
docker compose --env-file .runtime/demo.env -f deploy/compose.yml --profile documents up -d
```

新数据库使用 `docs/design/schema.sql`。对已有数据库应用迁移前，先备份并核对迁移记录；不要把备份加入仓库。
统一 Agent 的 `search_knowledge` 执行检索图，`read_section` 读取当前用户可读的原文。业务工具带 `mock` 标志，不代表真实资格或价格结果。
问答通过持久化 `QA_RUN` Job 执行，进度由 SSE 推送；模型答案经过引用与声明校验后展示。

Langfuse 默认关闭。在仓库根目录启动独立的合成观测环境：

```powershell
python scripts/langfuse_local.py init
docker compose --env-file .local/langfuse-eval.env -f deploy/langfuse.compose.yml up -d
```

配置保存在被忽略的 `.local/`。正文记录必须显式开启；关闭观测不会删除已发送的数据，数据保留期和权限由实例管理者负责。
在 backend 目录运行 `python scripts/export_rag_retrieval_graph.py --print` 可导出检索图。
默认测试不访问真实模型；带 `RAG_REAL_*` 开关的测试仅在使用者明确配置时运行。

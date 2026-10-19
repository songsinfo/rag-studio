import { test } from 'node:test'
import assert from 'node:assert/strict'
import { readFile } from 'node:fs/promises'
import ts from 'typescript'

const source = await readFile(new URL('../src/api.ts', import.meta.url), 'utf8')
const compiled = ts.transpileModule(source, {
  compilerOptions: { module: ts.ModuleKind.ESNext, target: ts.ScriptTarget.ES2022 },
}).outputText
const client = await import('data:text/javascript;base64,' + Buffer.from(compiled).toString('base64'))

test('frontend helper utilities: rows, keyOf, terminal', () => {
  // rows helper 统一解包数组、items、content、results 等常见响应结构，避免页面层重复兼容。
  assert.deepEqual(client.rows([1, 2, 3]), [1, 2, 3])
  assert.deepEqual(client.rows({ items: ['a', 'b'] }), ['a', 'b'])
  assert.deepEqual(client.rows({ content: ['c'] }), ['c'])
  assert.deepEqual(client.rows({ results: ['d'] }), ['d'])
  assert.deepEqual(client.rows(null), [])

  // keyOf 从不同实体中提取统一主键字段，兼容 id/runId/conversationId 等后端响应差异。
  assert.equal(client.keyOf({ id: 'id-1' }), 'id-1')
  assert.equal(client.keyOf({ documentId: 'doc-1' }), 'doc-1')
  assert.equal(client.keyOf({ runId: 'run-1' }), 'run-1')
  assert.equal(client.keyOf({ jobId: 'job-1' }), 'job-1')
  assert.equal(client.keyOf({ datasetId: 'ds-1' }), 'ds-1')
  assert.equal(client.keyOf({ versionId: 'ver-1' }), 'ver-1')
  assert.equal(client.keyOf({ conversationId: 'conv-1' }), 'conv-1')
  assert.equal(client.keyOf({ snapshotId: 'snap-1' }), 'snap-1')
  assert.equal(client.keyOf({ configId: 'cfg-1' }), 'cfg-1')
  assert.equal(client.keyOf({ caseId: 'case-1' }), 'case-1')
  assert.equal(client.keyOf({ resultId: 'res-1' }), 'res-1')

  // terminal helper 识别业务终态；只有终态才停止前端轮询，避免任务仍在运行时提前结束。
  assert.equal(client.terminal('QUEUED'), false)
  assert.equal(client.terminal('RUNNING'), false)
  assert.equal(client.terminal('PENDING'), false)
  for (const s of ['COMPLETED', 'FAILED', 'CANCELLED', 'VERIFICATION_FAILED', 'BUDGET_EXCEEDED']) {
    assert.equal(client.terminal(s), true)
  }
})

test('every frontend REST route correctly prefixes /api/v1 and formats headers', async () => {
  const routesToTest = [
    { path: '/auth/login', method: 'POST', body: { tenantCode: 'demo', loginName: 'test', password: '123' } },
    { path: '/auth/logout', method: 'POST' },
    { path: '/me', method: 'GET' },
    { path: '/knowledge-bases', method: 'GET' },
    { path: '/admin/knowledge-bases', method: 'POST', body: { name: 'kb-test', readers: [] } },
    { path: '/admin/knowledge-bases/kb-1/acl', method: 'GET' },
    { path: '/admin/knowledge-bases/kb-1/acl', method: 'PUT', body: { expectedVersion: 1, readers: [] } },
    { path: '/documents?kbId=kb-1', method: 'GET' },
    { path: '/admin/documents?status=TRASHED', method: 'GET' },
    { path: '/documents/doc-1', method: 'GET' },
    { path: '/documents/doc-1/versions/ver-1/preview', method: 'GET' },
    { path: '/admin/documents/doc-1/acl', method: 'GET' },
    { path: '/admin/documents/doc-1/acl', method: 'PUT', body: { expectedVersion: 1, mode: 'RESTRICT', readers: [] } },
    { path: '/admin/documents/doc-1/trash', method: 'POST', body: { expectedVersion: 1 } },
    { path: '/admin/documents/doc-1/restore', method: 'POST', body: { expectedVersion: 2 } },
    { path: '/admin/documents/doc-1/purge', method: 'POST', body: { expectedVersion: 3 } },
    { path: '/search?query=test&kbIds=kb-1', method: 'GET' },
    { path: '/conversations', method: 'GET' },
    { path: '/conversations', method: 'POST', body: { title: 'chat', scope: { mode: 'ALL', kbIds: [] } } },
    { path: '/conversations/cv-1', method: 'GET' },
    { path: '/conversations/cv-1', method: 'DELETE' },
    { path: '/conversations/cv-1/segments', method: 'POST', body: { scope: { mode: 'ALL', kbIds: [] } } },
    { path: '/conversations/cv-1/runs', method: 'POST', body: { question: 'q', mode: 'RAG', strict: false } },
    { path: '/runs/run-1', method: 'GET' },
    { path: '/runs/run-1/cancel', method: 'POST' },
    { path: '/runs/run-1/evidence/ev-1', method: 'GET' },
    { path: '/runs/run-1/feedback', method: 'PUT', body: { rating: 'USEFUL', comment: '', issueType: 'OTHER' } },
    { path: '/admin/jobs', method: 'GET' },
    { path: '/admin/jobs/job-1', method: 'GET' },
    { path: '/admin/jobs/job-1/retry', method: 'POST' },
    { path: '/admin/jobs/job-1/cancel', method: 'POST' },
    { path: '/admin/configs', method: 'GET' },
    { path: '/admin/configs', method: 'POST', body: { kind: 'RETRIEVAL', name: 'cfg', payload: {} } },
    { path: '/admin/configs/cfg-1/publish', method: 'POST' },
    { path: '/admin/feedback', method: 'GET' },
    { path: '/admin/feedback/fb-1', method: 'PATCH', body: { status: 'IN_PROGRESS', resolution: '' } },
    { path: '/admin/users', method: 'GET' },
    { path: '/admin/groups', method: 'GET' },
    { path: '/admin/groups/grp-1/members', method: 'PUT', body: { userIds: ['u-1'], expectedSecurityEpoch: 1 } },
    { path: '/admin/runs/run-1/trace', method: 'GET' },
    { path: '/admin/eval/datasets', method: 'GET' },
    { path: '/admin/eval/datasets', method: 'POST', body: { name: 'suite' } },
    { path: '/admin/eval/datasets/ds-1/versions', method: 'POST', body: { sourceVersionId: null } },
    { path: '/admin/eval/dataset-versions/ver-1/cases', method: 'GET' },
    { path: '/admin/eval/dataset-versions/ver-1/cases', method: 'POST', body: { cases: [] } },
    { path: '/admin/eval/dataset-versions/ver-1/freeze', method: 'POST' },
    { path: '/admin/eval/snapshots', method: 'GET' },
    { path: '/admin/eval/snapshots', method: 'POST', body: { name: 'snap', documentIds: [], approvedSyntheticOrRedacted: true } },
    { path: '/admin/eval/estimates', method: 'POST', body: { datasetVersionId: 'v1', snapshotId: 's1', configId: 'c1' } },
    { path: '/admin/eval/runs', method: 'POST', body: { datasetVersionId: 'v1', snapshotId: 's1', configId: 'c1', maxCostCny: 10, estimateId: 'est-1' } },
    { path: '/admin/eval/runs/r-1', method: 'GET' },
    { path: '/admin/eval/runs/r-1/results', method: 'GET' },
    { path: '/admin/eval/runs/r-1/cancel', method: 'POST' },
    { path: '/admin/eval/runs/r-1/exports', method: 'POST', body: { mode: 'REDACTED' } },
    { path: '/admin/eval/results/res-1/reviews', method: 'POST', body: { labels: {}, reason: '' } },
    { path: '/admin/eval/comparisons?runIds=r1,r2', method: 'GET' },
  ]

  let csrfRefreshed = false
  globalThis.fetch = async (url, options) => {
    if (url.endsWith('/csrf')) {
      csrfRefreshed = true
      return Response.json({ csrfToken: 'test-csrf-token' })
    }
    return Response.json({ status: 'OK', url, method: options?.method || 'GET' })
  }

  // 预先写入 CSRF 状态，聚焦测试 API 封装本身而不是重复覆盖 token 获取流程。
  await client.refreshCsrf()

  for (const item of routesToTest) {
    const res = await client.api(item.path, item.method, item.body)
    assert.equal(res.url, '/api/v1' + item.path, `Route mismatch for ${item.path}`)
    assert.equal(res.method, item.method, `Method mismatch for ${item.path}`)
  }
})

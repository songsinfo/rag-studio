// 前端统一 API 封装：集中处理 Cookie 会话、CSRF、幂等键、错误解析与鉴权过期事件。
export type Row = Record<string, any>
// CSRF token 缓存在内存中；页面刷新后重新获取，不写 localStorage，减少泄漏面。
let csrf = ''
let csrfHeader = 'X-CSRF-Token'
export class ApiError extends Error { constructor(public status: number, public code: string, message: string) { super(message) } }
export async function refreshCsrf() {
  // 写请求前按需获取最新 CSRF 配置，后端可自定义 header 名。
  const response = await fetch('/api/v1/auth/csrf', { credentials: 'include', cache: 'no-store' })
  if (!response.ok) throw new ApiError(response.status, 'CSRF_UNAVAILABLE', '无法获取安全令牌，请检查后端连接')
  const value = await response.json()
  csrf = value.token || value.csrfToken
  csrfHeader = value.headerName || 'X-CSRF-Token'
  if (!csrf) throw new Error('后端未返回 CSRF token')
}
export async function api(path: string, method = 'GET', body?: unknown): Promise<any> {
  const headers: Record<string, string> = {}
  // 所有非只读请求统一附带 CSRF + Idempotency-Key。
  // 每次用户操作生成幂等键；安全令牌刷新后的重试复用同一幂等键。
  if (!['GET', 'HEAD'].includes(method)) {
    if (!csrf) await refreshCsrf()
    headers[csrfHeader] = csrf
    headers['Idempotency-Key'] = crypto.randomUUID()
  }
  if (body !== undefined && !(body instanceof FormData)) headers['Content-Type'] = 'application/json'
  const options: RequestInit = { method, credentials: 'include', cache: 'no-store', headers, body: body === undefined ? undefined : body instanceof FormData ? body : JSON.stringify(body) }
  let response = await fetch('/api/v1' + path, options)
  // CSRF 校验在业务执行前拒绝请求，刷新后仅重试一次，避免令牌过期阻断问答。
  if (response.status === 403 && !['GET', 'HEAD'].includes(method)) {
    const rejection = await response.clone().json().catch(() => null)
    if (rejection?.code === 'CSRF_INVALID') {
      await refreshCsrf()
      headers[csrfHeader] = csrf
      response = await fetch('/api/v1' + path, options)
    }
  }
  const text = await response.text()
  let data: any
  try { data = text ? JSON.parse(text) : {} } catch { throw new ApiError(response.status, 'INVALID_RESPONSE', '后端返回了无法解析的响应') }
  if (!response.ok) {
    // 401 通过全局事件通知 App 统一回到登录态，而不是每个页面各自处理。
    if (response.status === 401) window.dispatchEvent(new Event('auth-expired'))
    throw new ApiError(response.status, data.code || 'HTTP_' + response.status, data.message || response.statusText)
  }
  return data
}
export const rows = (value: any): Row[] => Array.isArray(value) ? value : value?.items || value?.content || value?.results || []
export const pretty = (value: unknown) => JSON.stringify(value, null, 2)
export const keyOf = (row: Row) => row.id || row.documentId || row.runId || row.jobId || row.datasetId || row.versionId || row.conversationId || row.snapshotId || row.configId || row.caseId || row.resultId
export const terminal = (status: string) => !['QUEUED', 'RUNNING', 'PENDING'].includes(status)

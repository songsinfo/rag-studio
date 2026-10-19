<script setup lang="ts">
import { onMounted, onUnmounted, ref } from 'vue'
import { ElMessageBox } from 'element-plus'
import { api, rows, keyOf, terminal, type Row } from './api'
import JsonView from './JsonView.vue'
const conversations = ref<Row[]>([]), kbs = ref<Row[]>([]), conversation = ref<Row | null>(null), history = ref<Row[]>([]), selectedKbs = ref<string[]>([]), scopeMode = ref('ALL'), question = ref(''), strict = ref(false), current = ref<Row | null>(null), error = ref(''), busy = ref(false), evidence = ref<any>(null), evidenceOpen = ref(false), feedbackRun = ref(''), feedbackOpen = ref(false)
const feedback = ref({ rating: 'USEFUL', comment: '', issueType: 'OTHER' })
// 轮询只允许存在一个 timer；组件卸载后用 disposed 防止异步响应继续改状态。
let timer: ReturnType<typeof setTimeout> | undefined, stream: EventSource | undefined, disposed = false
function hasMockSource(run: Row) { return !!run.mock || (Array.isArray(run.evidence) && run.evidence.some((item: Row) => !!item.mock)) }
function stopUpdates() { clearTimeout(timer); stream?.close(); stream = undefined }
async function act(fn: () => Promise<void>) { error.value = ''; busy.value = true; try { await fn() } catch (e) { error.value = String(e) } finally { busy.value = false } }
async function refreshList() { conversations.value = rows(await api('/conversations')) }
// 打开会话时顺便恢复当前 segment scope，并检测是否存在尚未结束的 run。
async function open(id: string) { stopUpdates(); current.value = null; conversation.value = await api('/conversations/' + id); history.value = rows(conversation.value?.runs || conversation.value?.history || []); const segment = conversation.value?.currentSegment || conversation.value?.segments?.at(-1); scopeMode.value = segment?.scope?.mode || 'ALL'; selectedKbs.value = segment?.scope?.kbIds || []; const active = history.value.find(row => !terminal(row.status)); if (active) { current.value = active; watchRun() } }
async function create() { const result = await api('/conversations', 'POST', { title: question.value.slice(0, 60) || '新会话', scope: { mode: scopeMode.value, kbIds: selectedKbs.value } }); await refreshList(); await open(result.conversationId || result.id) }
// 修改知识库范围会创建新的 context segment；旧历史仍保留，但后续问题只使用新 scope。
async function segment() { await act(async () => {
  if (!conversation.value) return
  const previous = conversation.value.currentSegment?.scope || conversation.value.segments?.at(-1)?.scope || { mode: 'ALL', kbIds: [] }
  try {
    const scope = { mode: scopeMode.value, kbIds: [...selectedKbs.value] }
    const result = await api(`/conversations/${keyOf(conversation.value)}/segments`, 'POST', { scope })
    conversation.value.currentSegmentId = result.currentSegmentId || result.segmentId || result.id
    conversation.value.currentSegment = { ...result, scope }
  } catch (e) { scopeMode.value = previous.mode; selectedKbs.value = previous.kbIds; throw e }
}) }
// 普通 RAG 在响应后立即执行；Agent 仍使用 Durable Worker，前端订阅执行事件。
async function submit() { await act(async () => { if (!question.value.trim()) return; if (!conversation.value) await create(); const cv = conversation.value!; let segmentId = cv.currentSegmentId || cv.segmentId || cv.currentSegment?.id || cv.segments?.at(-1)?.id; if (!segmentId) { const s = await api(`/conversations/${keyOf(cv)}/segments`, 'POST', { scope: { mode: scopeMode.value, kbIds: selectedKbs.value } }); segmentId = s.currentSegmentId || s.segmentId || s.id; cv.currentSegmentId = segmentId } current.value = await api(`/conversations/${keyOf(cv)}/runs`, 'POST', { question: question.value, segmentId, strict: strict.value }); question.value = ''; watchRun() }) }
// SSE 传输实时阶段进度，断线时回退到状态查询；答案仅在通过 claim 验证后展示。
function watchRun() {
  stopUpdates()
  if (!current.value || disposed) return
  const runId = current.value.runId || current.value.id
  if (!runId) return
  if (typeof EventSource === 'undefined') { void poll(); return }
  const source = new EventSource('/api/v1/runs/' + encodeURIComponent(runId) + '/events', { withCredentials: true })
  stream = source
  source.addEventListener('run_event', (message: MessageEvent) => {
    if (disposed || stream !== source || !current.value) return
    try {
      const event = JSON.parse(message.data)
      current.value = { ...current.value, status: 'RUNNING', stage: event.stage || current.value.stage }
    } catch { /* 忽略无法解析的进度事件，最终状态仍从服务端读取。 */ }
  })
  source.addEventListener('done', () => { if (stream === source) { stopUpdates(); void poll() } })
  source.onerror = () => {
    if (stream === source) { stopUpdates(); void poll() }
  }
}
// 页面可见时高频轮询，后台标签页降低频率，减少无意义请求。
async function poll() { if (!current.value || disposed) return; try { const id = current.value.runId || current.value.id; const result = await api('/runs/' + id); if (disposed || !current.value || (current.value.runId || current.value.id) !== id) return; current.value = result; if (terminal(result.status)) { stopUpdates(); const cv = await api('/conversations/' + keyOf(conversation.value!)); history.value = rows(cv.runs || cv.history || []); current.value = null; return } timer = setTimeout(poll, document.hidden ? 5000 : 1000) } catch (e) {
    error.value = String(e)
    // 授权/网络错误后不自动无限重试；用户显式刷新即可恢复，避免错误风暴。
  } }
async function cancel() { await act(async () => { if (!current.value) return; await api('/runs/' + keyOf(current.value) + '/cancel', 'POST'); stopUpdates(); await open(keyOf(conversation.value!)) }) }
async function remove() { try { await ElMessageBox.confirm('清除当前会话和问答正文？活动任务会被取消。', '清除会话', { cancelButtonText: '取消' }) } catch { return } await act(async () => { await api('/conversations/' + keyOf(conversation.value!), 'DELETE'); conversation.value = null; history.value = []; current.value = null; stopUpdates(); await refreshList() }) }
async function source(run: Row, item: Row) { await act(async () => { evidence.value = await api(`/runs/${keyOf(run)}/evidence/${item.evidenceId || item.id}`); evidenceOpen.value = true }) }
async function saveFeedback() { await act(async () => { await api('/runs/' + feedbackRun.value + '/feedback', 'PUT', feedback.value); feedbackOpen.value = false }) }
onMounted(() => act(async () => { kbs.value = rows(await api('/knowledge-bases')); await refreshList() }))
onUnmounted(() => { disposed = true; stopUpdates() })
</script>
<template><div class="grid"><aside class="sidebar"><el-button type="primary" :disabled="!!current" @click="act(create)">新建会话</el-button><el-button v-for="item in conversations" :key="keyOf(item)" @click="act(() => open(keyOf(item)))">{{ item.title || keyOf(item) }}</el-button></aside><section><el-alert v-if="error" :title="error" type="error" :closable="false" /><div class="toolbar"><el-select v-model="scopeMode" :disabled="!!current || busy" @change="scopeMode === 'ALL' || selectedKbs.length ? segment() : undefined"><el-option value="ALL" label="全部可读知识库" /><el-option value="SELECTED" label="指定知识库" /></el-select><el-select v-if="scopeMode === 'SELECTED'" v-model="selectedKbs" multiple :disabled="!!current || busy" @change="segment"><el-option v-for="kb in kbs" :key="keyOf(kb)" :value="keyOf(kb)" :label="kb.name" /></el-select><el-button v-if="conversation" @click="act(() => open(keyOf(conversation!)))">刷新历史</el-button><el-button v-if="conversation" @click="remove">清除会话</el-button></div><p class="muted">切换资料范围会开启新的上下文段。历史与引用按当前权限重新检查。</p><div class="history"><el-empty v-if="!history.length" description="输入一个具体问题，开始查阅资料" /><el-card v-for="run in history" :key="keyOf(run)"><strong>{{ run.question }}</strong><el-alert v-if="run.visibility === 'HIDDEN'" title="此回答关联资料的访问权限已变化，内容暂不可见。" type="warning" :closable="false" /><template v-else><div class="toolbar"><el-tag>{{ run.status }}</el-tag><span class="muted">{{ run.terminalReason }}</span></div><p class="answer">{{ run.answer || '没有可展示的已验证答案。' }}</p><el-alert v-if="hasMockSource(run)" title="模拟结果，未查询真实业务系统" type="warning" :closable="false" /><p v-if="run.gaps?.length">请补充：{{ run.gaps.join('；') }}</p><div class="evidence"><el-button v-for="(item, index) in run.evidence || []" :key="item.evidenceId" size="small" @click="source(run, item)">{{ item.sourceType === 'TOOL' ? '模拟业务来源' : '文档引用' }} {{ Number(index) + 1 }} {{ item.mock ? '· 模拟结果，未查询真实业务系统' : '' }} {{ item.isCurrent === false ? '· 旧版' : '' }} {{ item.machineGenerated ? '· 机器识别' : '' }}</el-button></div><el-button v-if="run.answer" size="small" @click="feedbackRun = keyOf(run); feedbackOpen = true">反馈</el-button><el-collapse><el-collapse-item title="校验、告警与用量"><JsonView :value="{ verification: run.verification, warnings: run.warnings, usage: run.usageSummary }" /></el-collapse-item></el-collapse></template></el-card></div><el-alert v-if="current" :title="`${current.status} · ${current.stage || '排队处理中'}`" type="info" :closable="false"><el-button size="small" @click="cancel">取消运行</el-button></el-alert><el-form @submit.prevent="submit"><el-input v-model="question" type="textarea" :rows="3" maxlength="2000" placeholder="例如：订单补偿任务反复重试，应该按什么步骤排查？" :disabled="!!current" aria-label="问题" /><div class="toolbar"><el-tag>统一 Agent</el-tag><el-switch v-model="strict" active-text="严格证据验证" /><el-button native-type="submit" type="primary" :loading="busy" :disabled="!!current || !question.trim() || (scopeMode === 'SELECTED' && !selectedKbs.length)">提交问题</el-button></div></el-form></section><el-dialog v-model="evidenceOpen" title="来源详情" width="850px"><el-alert v-if="evidence?.mock" title="模拟结果，未查询真实业务系统" type="warning" :closable="false" /><JsonView :value="evidence" /></el-dialog><el-dialog v-model="feedbackOpen" title="回答反馈" width="500px"><el-form label-position="top"><el-form-item label="是否有用"><el-radio-group v-model="feedback.rating"><el-radio value="USEFUL">有用</el-radio><el-radio value="USELESS">无用</el-radio></el-radio-group></el-form-item><el-form-item label="问题类型"><el-select v-model="feedback.issueType"><el-option v-for="value in ['ANSWER_ERROR','CITATION_ERROR','OUTDATED','MISSING','PERMISSION','OTHER']" :key="value" :value="value" :label="value" /></el-select></el-form-item><el-form-item label="说明"><el-input v-model="feedback.comment" type="textarea" :rows="3" /></el-form-item></el-form><template #footer><el-button type="primary" @click="saveFeedback">提交反馈</el-button></template></el-dialog></div></template>

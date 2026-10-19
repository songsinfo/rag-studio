<script setup lang="ts">
import { onMounted, onUnmounted, ref } from 'vue'
import { api, refreshCsrf, type Row } from './api'
import Documents from './Documents.vue'
import Chat from './Chat.vue'
import Evaluation from './Evaluation.vue'
import Operations from './Operations.vue'
const me = ref<Row | null>(null), error = ref(''), busy = ref(false), tab = ref('chat')
const login = ref({ tenantCode: 'demo', loginName: '', password: '' })
async function signIn() { busy.value = true; error.value = ''; try { await refreshCsrf(); await api('/auth/login', 'POST', login.value); await refreshCsrf(); me.value = await api('/me'); login.value.password = '' } catch (e) { error.value = String(e) } finally { busy.value = false } }
async function signOut() { try { await api('/auth/logout', 'POST'); expired() } catch (e) { error.value = String(e) } }
function expired() { me.value = null }
onMounted(async () => { window.addEventListener('auth-expired', expired); try { me.value = await api('/me') } catch { /* Login is expected on a fresh browser. */ } })
onUnmounted(() => window.removeEventListener('auth-expired', expired))
</script>
<template>
  <main v-if="!me" class="login"><el-card><h1>企业知识库</h1><p>登录后，仅检索你有权阅读的资料。</p><el-form @submit.prevent="signIn" label-position="top"><el-form-item label="企业代码"><el-input v-model="login.tenantCode" required /></el-form-item><el-form-item label="账号"><el-input v-model="login.loginName" autocomplete="username" required /></el-form-item><el-form-item label="密码"><el-input v-model="login.password" type="password" autocomplete="current-password" show-password required /></el-form-item><el-alert v-if="error" :title="error" type="error" :closable="false" /><el-button native-type="submit" type="primary" :loading="busy">登录</el-button></el-form></el-card></main>
  <main v-else class="shell"><header><div><h1>企业知识库</h1><span>{{ me.displayName || me.loginName }} · {{ me.tenantCode || '内部工作空间' }}</span></div><el-button @click="signOut">退出登录</el-button></header><el-alert v-if="error" :title="error" type="error" @close="error = ''" /><el-tabs v-model="tab"><el-tab-pane label="问答" name="chat" /><el-tab-pane label="资料" name="documents" /><template v-if="me.isAdmin || me.admin || me.maintenanceAdmin || me.roles?.includes('ADMIN')"><el-tab-pane label="实验评估" name="evaluation" /><el-tab-pane label="运行管理" name="operations" /></template></el-tabs><Chat v-if="tab === 'chat'" /><Documents v-if="tab === 'documents'" :admin="!!(me.isAdmin || me.admin || me.maintenanceAdmin || me.roles?.includes('ADMIN'))" /><Evaluation v-if="tab === 'evaluation'" /><Operations v-if="tab === 'operations'" /></main>
</template>

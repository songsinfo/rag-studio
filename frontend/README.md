# RAG Studio Console

Vue 3 + TypeScript + Element Plus。开发代理请求 `127.0.0.1:8080` 的 `/api/v1`，保持同源 Cookie 与 CSRF；部署时将 `/api` 代理到后端。

```powershell
npm ci
npm run check
npm test
npm run build
npm run dev
```

使用本地初始化生成的演示账号。授权由后端校验，SSE 提供任务进度，终态展示经过校验的答案。
模拟工具结果明确显示为模拟结果，不查询真实业务系统。

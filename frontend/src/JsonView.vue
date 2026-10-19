<script setup lang="ts">defineProps<{ value: unknown }>()</script>
<template><pre class="json">{{ JSON.stringify(value, null, 2) }}</pre></template>

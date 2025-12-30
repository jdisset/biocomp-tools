import type {
  ComputeResponse,
  ConfigResponse,
  DiagramResponse,
  ParamGroupsResponse,
  StatusResponse,
} from '@/types/api'

const API_BASE = ''

async function fetchJson<T>(url: string, options?: RequestInit): Promise<T> {
  const response = await fetch(`${API_BASE}${url}`, {
    ...options,
    headers: {
      'Content-Type': 'application/json',
      ...options?.headers,
    },
  })
  if (!response.ok) {
    const error = await response.text()
    throw new Error(error || `HTTP ${response.status}`)
  }
  return response.json()
}

export async function fetchStatus(): Promise<StatusResponse> {
  return fetchJson('/status')
}

export async function fetchConfig(): Promise<ConfigResponse> {
  return fetchJson('/config')
}

export async function fetchParamGroups(): Promise<ParamGroupsResponse> {
  return fetchJson('/params/groups')
}

export async function updateParams(updates: Record<string, number | number[] | number[][]>): Promise<void> {
  await fetchJson('/params', {
    method: 'POST',
    body: JSON.stringify({ updates }),
  })
}

export async function compute(): Promise<ComputeResponse> {
  return fetchJson('/compute', { method: 'POST' })
}

export async function computeDetailed(includeContributions = false): Promise<ComputeResponse> {
  return fetchJson(`/compute/detailed?include_contributions=${includeContributions}`, {
    method: 'POST',
  })
}

export async function resetParams(seed?: number): Promise<void> {
  const url = seed !== undefined ? `/reset?seed=${seed}` : '/reset'
  await fetchJson(url, { method: 'POST' })
}

export async function fetchDiagram(plotType = 'all'): Promise<DiagramResponse> {
  return fetchJson(`/diagram?plot_type=${plotType}`)
}

export async function exportParams(): Promise<Record<string, number | number[] | number[][]>> {
  return fetchJson('/export')
}

export async function importParams(params: Record<string, number | number[] | number[][]>): Promise<void> {
  await fetchJson('/import', {
    method: 'POST',
    body: JSON.stringify(params),
  })
}

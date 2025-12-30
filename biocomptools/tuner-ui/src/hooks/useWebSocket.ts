import { useEffect, useRef, useCallback } from 'react'
import { useTunerStore } from '@/store/tunerStore'
import type { WSMessage, WSResponse, ComputeResponse } from '@/types/api'

export function useWebSocket() {
  const ws = useRef<WebSocket | null>(null)
  const reconnectAttempts = useRef(0)
  const maxReconnectAttempts = 5
  const { updateFromCompute } = useTunerStore()

  const connect = useCallback(() => {
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:'
    const wsUrl = `${protocol}//${window.location.host}/ws`

    ws.current = new WebSocket(wsUrl)

    ws.current.onopen = () => {
      reconnectAttempts.current = 0
    }

    ws.current.onmessage = (event) => {
      try {
        const msg: WSResponse = JSON.parse(event.data)
        if (msg.type === 'compute_result') {
          updateFromCompute(msg.data as ComputeResponse)
        }
      } catch {
        // ignore parse errors
      }
    }

    ws.current.onerror = () => {
      // will trigger onclose
    }

    ws.current.onclose = () => {
      if (reconnectAttempts.current < maxReconnectAttempts) {
        reconnectAttempts.current++
        setTimeout(connect, 1000 * reconnectAttempts.current)
      }
    }
  }, [updateFromCompute])

  useEffect(() => {
    connect()
    return () => {
      ws.current?.close()
    }
  }, [connect])

  const sendMessage = useCallback((message: WSMessage) => {
    if (ws.current?.readyState === WebSocket.OPEN) {
      ws.current.send(JSON.stringify(message))
    }
  }, [])

  const sendParamUpdate = useCallback(
    (path: string, value: number) => {
      sendMessage({ action: 'update_param', path, value })
    },
    [sendMessage]
  )

  const sendBatchUpdate = useCallback(
    (updates: Record<string, number>) => {
      sendMessage({ action: 'update_params_batch', updates })
    },
    [sendMessage]
  )

  const requestCompute = useCallback(() => {
    sendMessage({ action: 'compute' })
  }, [sendMessage])

  const requestReset = useCallback(
    (seed?: number) => {
      sendMessage({ action: 'reset', seed })
    },
    [sendMessage]
  )

  return {
    sendParamUpdate,
    sendBatchUpdate,
    requestCompute,
    requestReset,
    isConnected: ws.current?.readyState === WebSocket.OPEN,
  }
}

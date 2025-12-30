import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { useEffect, useState } from 'react'
import { fetchStatus, fetchParamGroups, compute } from '@/api/client'
import { useTunerStore } from '@/store/tunerStore'
import { AppLayout } from '@/components/layout/AppLayout'
import { HeatmapView } from '@/components/visualization/HeatmapView'
import { LossHistory } from '@/components/visualization/LossHistory'
import { LossDisplay } from '@/components/losses/LossDisplay'
import { ParamsPanel } from '@/components/parameters/ParamsPanel'
import { NetworkDiagram } from '@/components/visualization/NetworkDiagram'

const queryClient = new QueryClient()

function LoadingScreen() {
  return (
    <div className="min-h-screen flex items-center justify-center bg-background">
      <div className="text-center">
        <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-primary mx-auto mb-4" />
        <p className="text-muted-foreground">Loading biocomp-tuner...</p>
      </div>
    </div>
  )
}

function ErrorScreen({ error }: { error: string }) {
  return (
    <div className="min-h-screen flex items-center justify-center bg-background">
      <div className="text-center p-8 rounded-lg border border-destructive/50 bg-destructive/10">
        <h2 className="text-xl font-semibold text-destructive mb-2">Error</h2>
        <p className="text-muted-foreground">{error}</p>
      </div>
    </div>
  )
}

function TunerApp() {
  const [isLoading, setIsLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const { setInitialized, setParamGroups, updateFromCompute } = useTunerStore()

  useEffect(() => {
    async function initialize() {
      try {
        const status = await fetchStatus()
        if (!status.initialized) {
          throw new Error('Session not initialized')
        }
        setInitialized(status.network_name || 'Unknown', status.grid_resolution)

        const paramsResp = await fetchParamGroups()
        setParamGroups(paramsResp.groups)

        const result = await compute()
        updateFromCompute(result)

        setIsLoading(false)
      } catch (e) {
        setError(e instanceof Error ? e.message : 'Failed to initialize')
        setIsLoading(false)
      }
    }
    initialize()
  }, [setInitialized, setParamGroups, updateFromCompute])

  if (isLoading) return <LoadingScreen />
  if (error) return <ErrorScreen error={error} />

  return (
    <AppLayout>
      <div className="flex-1 flex flex-col gap-4 p-4 overflow-auto">
        <div className="grid grid-cols-2 gap-4 shrink-0">
          <HeatmapView type="target" />
          <HeatmapView type="prediction" />
        </div>
        <div className="flex-1 min-h-0">
          <LossHistory />
        </div>
      </div>

      <aside className="w-96 border-l border-border flex flex-col overflow-hidden">
        <LossDisplay />
        <ParamsPanel />
      </aside>

      <NetworkDiagram />
    </AppLayout>
  )
}

export default function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <TunerApp />
    </QueryClientProvider>
  )
}

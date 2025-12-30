import { useTunerStore } from '@/store/tunerStore'

interface AppLayoutProps {
  children: React.ReactNode
}

export function AppLayout({ children }: AppLayoutProps) {
  const { networkName, gridResolution } = useTunerStore()

  return (
    <div className="min-h-screen flex flex-col bg-background">
      <header className="h-12 border-b border-border px-4 flex items-center justify-between">
        <div className="flex items-center gap-4">
          <h1 className="text-lg font-semibold text-primary">biocomp-tuner</h1>
          {networkName && (
            <span className="text-sm text-muted-foreground">{networkName}</span>
          )}
        </div>
        <div className="text-sm text-muted-foreground">
          {gridResolution[0]} x {gridResolution[1]}
        </div>
      </header>
      <main className="flex-1 flex overflow-hidden">
        {children}
      </main>
    </div>
  )
}

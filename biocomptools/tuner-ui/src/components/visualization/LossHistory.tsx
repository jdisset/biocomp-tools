import { useMemo } from 'react'
import Plot from 'react-plotly.js'
import { Trash2 } from 'lucide-react'
import { useTunerStore } from '@/store/tunerStore'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Button } from '@/components/ui/button'

export function LossHistory() {
  const { lossHistory, clearLossHistory } = useTunerStore()

  const yRange = useMemo(() => {
    if (lossHistory.length < 2) return undefined

    const allValues = lossHistory.flatMap((h) => [
      h.losses.total,
      h.losses.sinkhorn,
      h.losses.lncc,
      h.losses.mse,
      h.losses.simse,
    ]).filter((v) => v > 0 && isFinite(v))

    if (allValues.length === 0) return undefined

    const minVal = Math.min(...allValues)
    const maxVal = Math.max(...allValues)

    // Add padding in log space (0.3 decades)
    const logMin = Math.log10(minVal) - 0.3
    const logMax = Math.log10(maxVal) + 0.3

    return [logMin, logMax]
  }, [lossHistory])

  if (lossHistory.length < 2) {
    return (
      <Card className="flex items-center justify-center h-full">
        <p className="text-sm text-muted-foreground">
          Adjust parameters to see loss history
        </p>
      </Card>
    )
  }

  const x = lossHistory.map((_, i) => i)

  return (
    <Card className="h-full flex flex-col">
      <CardHeader className="py-2 px-3 flex-row items-center justify-between shrink-0">
        <CardTitle className="text-sm">Loss History</CardTitle>
        <Button variant="ghost" size="icon" className="h-6 w-6" onClick={clearLossHistory}>
          <Trash2 className="h-4 w-4" />
        </Button>
      </CardHeader>
      <CardContent className="p-0 flex-1 min-h-0">
        <Plot
          data={[
            {
              x,
              y: lossHistory.map((h) => h.losses.total),
              type: 'scatter',
              mode: 'lines',
              name: 'Total',
              line: { color: '#4fc3f7', width: 2 },
            },
            {
              x,
              y: lossHistory.map((h) => h.losses.sinkhorn),
              type: 'scatter',
              mode: 'lines',
              name: 'Sinkhorn',
              line: { color: '#ff9800', dash: 'dot' },
            },
            {
              x,
              y: lossHistory.map((h) => h.losses.lncc),
              type: 'scatter',
              mode: 'lines',
              name: 'LNCC',
              line: { color: '#4caf50', dash: 'dot' },
            },
            {
              x,
              y: lossHistory.map((h) => h.losses.mse),
              type: 'scatter',
              mode: 'lines',
              name: 'MSE',
              line: { color: '#e91e63', dash: 'dot' },
            },
            {
              x,
              y: lossHistory.map((h) => h.losses.simse),
              type: 'scatter',
              mode: 'lines',
              name: 'SIMSE',
              line: { color: '#9c27b0', dash: 'dot' },
            },
          ] as any}
          layout={{
            autosize: true,
            margin: { t: 30, b: 40, l: 50, r: 20 },
            paper_bgcolor: 'transparent',
            plot_bgcolor: 'transparent',
            font: { color: '#a1a1aa', size: 10 },
            legend: { x: 0, y: 1.05, orientation: 'h' },
            xaxis: { title: { text: 'Updates' }, gridcolor: 'rgba(255,255,255,0.1)' },
            yaxis: { title: { text: 'Loss' }, type: 'log', range: yRange, autorange: !yRange, gridcolor: 'rgba(255,255,255,0.1)' },
          } as any}
          config={{ displayModeBar: false, responsive: true }}
          style={{ width: '100%', height: '100%' }}
        />
      </CardContent>
    </Card>
  )
}

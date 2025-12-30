import { useMemo } from 'react'
import Plot from 'react-plotly.js'
import { useTunerStore } from '@/store/tunerStore'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { formatNumber } from '@/lib/utils'

interface HeatmapViewProps {
  type: 'target' | 'prediction' | 'diff'
}

export function HeatmapView({ type }: HeatmapViewProps) {
  const { yTarget, yPred, yDiff, xLattice, losses } = useTunerStore()

  const data = type === 'target' ? yTarget : type === 'prediction' ? yPred : yDiff

  // Compute shared color range for target and prediction
  const { zmin, zmax } = useMemo(() => {
    if (type === 'diff') return { zmin: undefined, zmax: undefined }

    let min = Infinity
    let max = -Infinity

    for (const grid of [yTarget, yPred]) {
      if (!grid) continue
      for (const row of grid) {
        for (const val of row) {
          if (val < min) min = val
          if (val > max) max = val
        }
      }
    }

    return { zmin: min === Infinity ? undefined : min, zmax: max === -Infinity ? undefined : max }
  }, [yTarget, yPred, type])

  if (!data) {
    return (
      <Card className="flex items-center justify-center">
        <p className="text-sm text-muted-foreground">Loading...</p>
      </Card>
    )
  }

  const colorscale = type === 'diff' ? 'RdBu' : 'Viridis'

  // Extract axis values from lattice (assumes lattice is [N, 2] with [x, y] pairs)
  // Get unique x values (first column) and y values (second column)
  let xVals: number[] | undefined
  let yVals: number[] | undefined
  if (xLattice && xLattice.length > 0) {
    const gridSize = Math.sqrt(xLattice.length)
    if (Number.isInteger(gridSize)) {
      // Lattice is flattened grid, extract unique x and y values
      xVals = xLattice.slice(0, gridSize).map((p) => p[0])
      yVals = []
      for (let i = 0; i < gridSize; i++) {
        yVals.push(xLattice[i * gridSize][1])
      }
    }
  }

  const title =
    type === 'target'
      ? 'Target'
      : type === 'prediction'
        ? `Prediction (Loss: ${formatNumber(losses.total)})`
        : 'Difference (Pred - Target)'

  return (
    <Card>
      <CardHeader className="py-2 px-3">
        <CardTitle className="text-sm">{title}</CardTitle>
      </CardHeader>
      <CardContent className="p-0">
        <Plot
          data={[
            {
              z: data,
              x: xVals,
              y: yVals,
              type: 'heatmap',
              colorscale,
              showscale: type === 'prediction',
              zmin: type !== 'diff' ? zmin : undefined,
              zmax: type !== 'diff' ? zmax : undefined,
            },
          ] as any}
          layout={{
            autosize: true,
            height: 280,
            margin: { t: 10, b: 40, l: 50, r: 10 },
            paper_bgcolor: 'transparent',
            plot_bgcolor: 'transparent',
            font: { color: '#a1a1aa', size: 10 },
            xaxis: { title: { text: 'x' } },
            yaxis: { title: { text: 'y' }, scaleanchor: 'x', scaleratio: 1 },
          } as any}
          style={{ width: '100%' }}
          config={{ displayModeBar: false, responsive: true }}
        />
      </CardContent>
    </Card>
  )
}

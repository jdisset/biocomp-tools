import { RefreshCw, RotateCcw, Download, Upload } from 'lucide-react'
import { useTunerStore } from '@/store/tunerStore'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { cn, formatNumber, getLossColor } from '@/lib/utils'
import { compute, resetParams, exportParams, importParams } from '@/api/client'

function LossCard({
  label,
  value,
  isTotal = false,
}: {
  label: string
  value: number
  isTotal?: boolean
}) {
  return (
    <div className="flex items-center justify-between py-1">
      <span className="text-sm text-muted-foreground">{label}</span>
      <span className={cn('text-sm font-mono', isTotal && getLossColor(value))}>
        {formatNumber(value)}
      </span>
    </div>
  )
}

export function LossDisplay() {
  const { losses, penalties, isComputing, updateFromCompute, setIsComputing } = useTunerStore()

  const handleCompute = async () => {
    setIsComputing(true)
    try {
      const result = await compute()
      updateFromCompute(result)
    } finally {
      setIsComputing(false)
    }
  }

  const handleReset = async () => {
    setIsComputing(true)
    try {
      await resetParams()
      const result = await compute()
      updateFromCompute(result)
    } finally {
      setIsComputing(false)
    }
  }

  const handleExport = async () => {
    const params = await exportParams()
    const blob = new Blob([JSON.stringify(params, null, 2)], { type: 'application/json' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = 'tuner_params.json'
    a.click()
    URL.revokeObjectURL(url)
  }

  const handleImport = () => {
    const input = document.createElement('input')
    input.type = 'file'
    input.accept = '.json'
    input.onchange = async (e) => {
      const file = (e.target as HTMLInputElement).files?.[0]
      if (!file) return
      const text = await file.text()
      const params = JSON.parse(text)
      await importParams(params)
      const result = await compute()
      updateFromCompute(result)
    }
    input.click()
  }

  return (
    <Card className="rounded-none border-0 border-b">
      <CardHeader className="py-3 px-4">
        <CardTitle className="text-sm flex items-center justify-between">
          <span>Losses</span>
          <div className="flex gap-1">
            <Button
              variant="ghost"
              size="icon"
              className="h-6 w-6"
              onClick={handleCompute}
              disabled={isComputing}
            >
              <RefreshCw className={cn('h-3 w-3', isComputing && 'animate-spin')} />
            </Button>
            <Button variant="ghost" size="icon" className="h-6 w-6" onClick={handleReset}>
              <RotateCcw className="h-3 w-3" />
            </Button>
            <Button variant="ghost" size="icon" className="h-6 w-6" onClick={handleExport}>
              <Download className="h-3 w-3" />
            </Button>
            <Button variant="ghost" size="icon" className="h-6 w-6" onClick={handleImport}>
              <Upload className="h-3 w-3" />
            </Button>
          </div>
        </CardTitle>
      </CardHeader>
      <CardContent className="py-2 px-4">
        <LossCard label="Total" value={losses.total} isTotal />
        <LossCard label="Sinkhorn" value={losses.sinkhorn} />
        <LossCard label="LNCC" value={losses.lncc} />
        <LossCard label="MSE" value={losses.mse} />
        <LossCard label="SIMSE" value={losses.simse} />

        <div className="mt-3 pt-3 border-t border-border">
          <p className="text-xs text-muted-foreground mb-2">Penalties</p>
          <LossCard label="TU Count" value={penalties.tucount} />
          <LossCard label="Spread" value={penalties.spread} />
        </div>
      </CardContent>
    </Card>
  )
}

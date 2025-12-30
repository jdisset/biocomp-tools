import { useCallback } from 'react'
import { Minimize2 } from 'lucide-react'
import { useTunerStore } from '@/store/tunerStore'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Slider } from '@/components/ui/slider'
import { Input } from '@/components/ui/input'
import { Button } from '@/components/ui/button'
import { updateParams, compute } from '@/api/client'
import { useDebounce } from '@/hooks/useDebounce'
import type { ParamGroup } from '@/types/api'

interface RatioGroupProps {
  group: ParamGroup
}

export function RatioGroup({ group }: RatioGroupProps) {
  const { params, setParam, updateFromCompute, setIsComputing } = useTunerStore()

  const debouncedUpdate = useDebounce(async (updates: Record<string, number>) => {
    setIsComputing(true)
    try {
      await updateParams(updates)
      const result = await compute()
      updateFromCompute(result)
    } finally {
      setIsComputing(false)
    }
  }, 100)

  const handleChange = useCallback(
    (path: string, value: number) => {
      setParam(path, value)
      debouncedUpdate({ [path]: value })
    },
    [setParam, debouncedUpdate]
  )

  const handleNormalize = useCallback(() => {
    const currentValues: Record<string, number> = {}
    for (const param of group.params) {
      const storeVal = params[param.path]
      currentValues[param.path] = typeof storeVal === 'number' ? storeVal : 1
    }

    const minVal = Math.min(...Object.values(currentValues))
    if (minVal <= 0) return

    const normalized: Record<string, number> = {}
    for (const [path, val] of Object.entries(currentValues)) {
      normalized[path] = val / minVal
    }

    for (const [p, v] of Object.entries(normalized)) {
      setParam(p, v)
    }
    debouncedUpdate(normalized)
  }, [group.params, params, setParam, debouncedUpdate])

  return (
    <Card className="mb-2">
      <CardHeader className="py-2 px-3">
        <CardTitle className="text-xs flex items-center justify-between">
          <span>{group.group_name}</span>
          <div className="flex items-center gap-2">
            <span className="text-muted-foreground">{group.params.length} TUs</span>
            <Button
              variant="ghost"
              size="icon"
              className="h-5 w-5"
              onClick={handleNormalize}
              title="Normalize (set min to 1)"
            >
              <Minimize2 className="h-3 w-3" />
            </Button>
          </div>
        </CardTitle>
      </CardHeader>
      <CardContent className="py-2 px-3">
        <div className="grid grid-cols-1 gap-2">
          {group.params.map((param) => {
            const value = (params[param.path] as number) ?? 0
            return (
              <div key={param.path} className="flex items-center gap-2">
                <span className="text-xs w-20 truncate" title={param.tu_name || param.display_name}>
                  {param.tu_name || param.display_name}
                </span>
                <Slider
                  className="flex-1"
                  min={0.01}
                  max={150}
                  step={0.1}
                  value={[Math.max(0.01, value)]}
                  onValueChange={([v]) => handleChange(param.path, v)}
                />
                <Input
                  type="number"
                  className="w-16 h-7 text-xs px-2"
                  value={value.toFixed(2)}
                  step={0.1}
                  onChange={(e) => handleChange(param.path, parseFloat(e.target.value) || 0)}
                />
              </div>
            )
          })}
        </div>
      </CardContent>
    </Card>
  )
}

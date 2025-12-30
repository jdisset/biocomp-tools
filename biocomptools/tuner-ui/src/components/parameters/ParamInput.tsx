import { useCallback } from 'react'
import { useTunerStore } from '@/store/tunerStore'
import { Input } from '@/components/ui/input'
import { Slider } from '@/components/ui/slider'
import { updateParams, compute } from '@/api/client'
import { useDebounce } from '@/hooks/useDebounce'
import type { ParamDescriptor } from '@/types/api'

interface ParamInputProps {
  param: ParamDescriptor
}

export function ParamInput({ param }: ParamInputProps) {
  const { params, setParam, updateFromCompute, setIsComputing } = useTunerStore()
  const value = params[param.path]

  const debouncedUpdate = useDebounce(async (path: string, v: number | number[] | number[][]) => {
    setIsComputing(true)
    try {
      await updateParams({ [path]: v })
      const result = await compute()
      updateFromCompute(result)
    } finally {
      setIsComputing(false)
    }
  }, 150)

  const handleChange = useCallback(
    (newValue: number | number[] | number[][]) => {
      setParam(param.path, newValue)
      debouncedUpdate(param.path, newValue)
    },
    [param.path, setParam, debouncedUpdate]
  )

  const isScalar = param.shape.length === 0 || (param.shape.length === 1 && param.shape[0] === 1)
  const useSlider = param.ui_type === 'slider'

  if (isScalar) {
    const scalarValue: number = typeof value === 'number'
      ? value
      : Array.isArray(value) && typeof value[0] === 'number'
        ? value[0]
        : 0

    if (useSlider) {
      return (
        <div className="flex items-center gap-2">
          <span className="text-xs w-28 truncate" title={param.display_name}>
            {param.display_name}
          </span>
          <Slider
            className="flex-1"
            min={param.min_value ?? 0}
            max={param.max_value ?? 10}
            step={param.step ?? 0.01}
            value={[scalarValue]}
            onValueChange={([v]) => handleChange(v)}
          />
          <Input
            type="number"
            className="w-16 h-7 text-xs px-2"
            value={scalarValue.toFixed(2)}
            step={param.step ?? 0.01}
            onChange={(e) => handleChange(parseFloat(e.target.value) || 0)}
          />
        </div>
      )
    }

    return (
      <div className="flex items-center gap-2">
        <span className="text-xs flex-1 truncate" title={param.display_name}>
          {param.display_name}
        </span>
        <Input
          type="number"
          className="w-24 h-7 text-xs"
          value={scalarValue}
          step={param.step}
          min={param.min_value}
          max={param.max_value}
          onChange={(e) => handleChange(parseFloat(e.target.value) || 0)}
        />
      </div>
    )
  }

  return (
    <div className="space-y-1">
      <span className="text-xs text-muted-foreground">{param.display_name}</span>
      <div className="text-xs text-muted-foreground">
        Shape: [{param.shape.join(', ')}]
      </div>
    </div>
  )
}

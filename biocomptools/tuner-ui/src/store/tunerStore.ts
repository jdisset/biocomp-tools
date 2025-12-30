import { create } from 'zustand'
import { devtools } from 'zustand/middleware'
import type { ComputeResponse, LossValues, ParamGroup, PenaltyValues } from '@/types/api'

export interface LossHistoryEntry {
  timestamp: number
  losses: LossValues
  penalties: PenaltyValues
}

interface TunerState {
  isInitialized: boolean
  networkName: string | null
  gridResolution: [number, number]

  params: Record<string, number | number[] | number[][]>
  paramGroups: ParamGroup[]

  yPred: number[][] | null
  yTarget: number[][] | null
  yDiff: number[][] | null
  xLattice: number[][] | null
  losses: LossValues
  penalties: PenaltyValues

  lossHistory: LossHistoryEntry[]

  isComputing: boolean
  selectedView: 'target' | 'prediction' | 'diff'
  showDiagram: boolean

  setParam: (path: string, value: number | number[] | number[][]) => void
  setParams: (updates: Record<string, number | number[] | number[][]>) => void
  setParamGroups: (groups: ParamGroup[]) => void
  updateFromCompute: (result: ComputeResponse) => void
  addToLossHistory: (entry: LossHistoryEntry) => void
  clearLossHistory: () => void
  setIsComputing: (value: boolean) => void
  setSelectedView: (view: 'target' | 'prediction' | 'diff') => void
  setShowDiagram: (show: boolean) => void
  setInitialized: (name: string, resolution: [number, number]) => void
}

const defaultLosses: LossValues = {
  total: 0,
  sinkhorn: 0,
  lncc: 0,
  mse: 0,
  simse: 0,
}

const defaultPenalties: PenaltyValues = {
  tucount: 0,
  spread: 0,
}

export const useTunerStore = create<TunerState>()(
  devtools(
    (set) => ({
      isInitialized: false,
      networkName: null,
      gridResolution: [32, 32],
      params: {},
      paramGroups: [],
      yPred: null,
      yTarget: null,
      yDiff: null,
      xLattice: null,
      losses: defaultLosses,
      penalties: defaultPenalties,
      lossHistory: [],
      isComputing: false,
      selectedView: 'prediction',
      showDiagram: false,

      setParam: (path, value) => {
        set((state) => ({
          params: { ...state.params, [path]: value },
        }))
      },

      setParams: (updates) => {
        set((state) => ({
          params: { ...state.params, ...updates },
        }))
      },

      setParamGroups: (groups) => {
        const params: Record<string, number | number[] | number[][]> = {}
        for (const group of groups) {
          for (const param of group.params) {
            params[param.path] = param.current_value
          }
        }
        set({ paramGroups: groups, params })
      },

      updateFromCompute: (result) => {
        const entry: LossHistoryEntry = {
          timestamp: Date.now(),
          losses: result.losses,
          penalties: result.penalties,
        }

        set((state) => ({
          yPred: result.Y_pred,
          yTarget: result.Y_target,
          yDiff: result.Y_diff,
          xLattice: result.X_lattice,
          losses: result.losses,
          penalties: result.penalties,
          lossHistory: [...state.lossHistory.slice(-99), entry],
          isComputing: false,
        }))
      },

      addToLossHistory: (entry) => {
        set((state) => ({
          lossHistory: [...state.lossHistory.slice(-99), entry],
        }))
      },

      clearLossHistory: () => {
        set({ lossHistory: [] })
      },

      setIsComputing: (value) => {
        set({ isComputing: value })
      },

      setSelectedView: (view) => {
        set({ selectedView: view })
      },

      setShowDiagram: (show) => {
        set({ showDiagram: show })
      },

      setInitialized: (name, resolution) => {
        set({
          isInitialized: true,
          networkName: name,
          gridResolution: resolution,
        })
      },
    }),
    { name: 'tuner-store' }
  )
)

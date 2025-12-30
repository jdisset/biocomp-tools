export interface StatusResponse {
  initialized: boolean
  network_name: string | null
  scaffold_name: string | null
  target_name: string | null
  model_name: string | null
  grid_resolution: [number, number]
}

export interface ParamDescriptor {
  path: string
  display_name: string
  shape: number[]
  category: string
  current_value: number | number[] | number[][]
  min_value?: number
  max_value?: number
  layer_name?: string
  param_name?: string
  cotx_group?: string
  tu_name?: string
  ui_type: 'number' | 'slider' | 'dropdown'
  step: number
}

export interface ParamGroup {
  group_id: string
  group_name: string
  category: string
  params: ParamDescriptor[]
  is_ratio_group: boolean
  cotx_name?: string
  tu_names?: string[]
}

export interface ParamGroupsResponse {
  groups: ParamGroup[]
  total_count: number
}

export interface LossValues {
  total: number
  sinkhorn: number
  lncc: number
  mse: number
  simse: number
  spectral?: number
}

export interface PenaltyValues {
  tucount: number
  spread: number
}

export interface ComputeResponse {
  Y_pred: number[][]
  Y_target: number[][]
  Y_diff: number[][]
  X_lattice: number[][]
  losses: LossValues
  penalties: PenaltyValues
  loss_contributions?: Record<string, number[][]>
}

export interface DiagramResponse {
  svg: string
  plot_type: string
}

export interface ConfigResponse {
  model_name: string | null
  scaffold_name: string | null
  target_name: string | null
  grid_resolution: [number, number]
  loss_weights: {
    sinkhorn: number
    lncc: number
    mse: number
    simse: number
  }
}

export interface DesignCompareResponse {
  final_loss: number
  loss_history: Array<{
    step: number
    loss: number
    sinkhorn?: number
    lncc?: number
  }>
}

export interface WSMessage {
  action: 'update_param' | 'update_params_batch' | 'compute' | 'reset'
  path?: string
  value?: number
  updates?: Record<string, number>
  seed?: number
}

export interface WSResponse {
  type: 'compute_result' | 'error'
  data: ComputeResponse | { message: string }
}

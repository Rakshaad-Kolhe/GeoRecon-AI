export type DisplayMode = 'cloud' | 'mesh' | 'both'
export type ColorMode = 'rgb' | 'conf' | 'height'
export type MaterialMode = 'photo' | 'shaded'
export type DetailLevel = 'standard' | 'high'

export interface ViewState {
  display: DisplayMode
  /** unset until the default is picked from meta.triangles on first load. */
  displayAuto: boolean
  colorMode: ColorMode
  /** multiplier 0.5-3x on the world-space base point size (1.5x median spacing). */
  pointSize: number
  showTrajectory: boolean
  material: MaterialMode
  detail: DetailLevel
}

export const DEFAULT_VIEW_STATE: ViewState = {
  display: 'both',
  displayAuto: true,
  colorMode: 'rgb',
  pointSize: 1,
  showTrajectory: true,
  material: 'photo',
  detail: 'standard',
}

export interface CameraApi {
  reset: () => void
  top: () => void
  oblique: () => void
}

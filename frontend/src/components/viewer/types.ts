export type DisplayMode = 'cloud' | 'mesh' | 'both'
export type ColorMode = 'rgb' | 'conf' | 'height'
export type MaterialMode = 'photo' | 'shaded'
export type DetailLevel = 'standard' | 'high'
export type BackgroundMode = 'dark' | 'grey' | 'light'

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
  /** canvas clear colour toggle for colour evaluation. */
  background: BackgroundMode
  /** display-only linear brightness multiplier applied to vertex colours (0.6–1.6). */
  exposure: number
}

export const DEFAULT_VIEW_STATE: ViewState = {
  display: 'both',
  displayAuto: true,
  colorMode: 'rgb',
  pointSize: 1,
  showTrajectory: true,
  material: 'photo',
  detail: 'standard',
  background: 'dark',
  exposure: 1.0,
}

export interface CameraApi {
  reset: () => void
  top: () => void
  oblique: () => void
}

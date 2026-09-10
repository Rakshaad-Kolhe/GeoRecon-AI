export type DisplayMode = 'cloud' | 'mesh' | 'both'
export type ColorMode = 'rgb' | 'conf' | 'height'

export interface ViewState {
  display: DisplayMode
  colorMode: ColorMode
  pointSize: number
  showTrajectory: boolean
}

export const DEFAULT_VIEW_STATE: ViewState = {
  display: 'both',
  colorMode: 'rgb',
  pointSize: 1.5,
  showTrajectory: true,
}

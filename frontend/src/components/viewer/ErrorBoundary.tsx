import { Component, type ErrorInfo, type ReactNode } from 'react'

interface Props {
  children: ReactNode
  onError?: (message: string) => void
}

interface State {
  failed: boolean
}

export class ErrorBoundary extends Component<Props, State> {
  state: State = { failed: false }

  static getDerivedStateFromError(): State {
    return { failed: true }
  }

  componentDidCatch(error: Error, info: ErrorInfo) {
    this.props.onError?.(error.message || String(error))
    if (import.meta.env.DEV) console.error('viewer error', error, info)
  }

  render() {
    if (this.state.failed) return null
    return this.props.children
  }
}

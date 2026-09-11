import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import { createBrowserRouter, RouterProvider } from 'react-router-dom'
import { AppLayout } from './components/AppLayout'
import './index.css'
import { ColorTest } from './pages/ColorTest'
import { JobDetail } from './pages/JobDetail'
import { JobsList } from './pages/JobsList'
import { NewJob } from './pages/NewJob'

const router = createBrowserRouter([
  {
    element: <AppLayout />,
    children: [
      { path: '/', element: <JobsList /> },
      { path: '/new', element: <NewJob /> },
      { path: '/jobs/:id', element: <JobDetail /> },
      { path: '/colortest', element: <ColorTest /> },
    ],
  },
])

const queryClient = new QueryClient({
  defaultOptions: {
    queries: { refetchOnWindowFocus: false, retry: false },
  },
})

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <QueryClientProvider client={queryClient}>
      <RouterProvider router={router} />
    </QueryClientProvider>
  </StrictMode>,
)

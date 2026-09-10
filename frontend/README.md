# GeoRecon AI — frontend

Mission-console UI for the drone-video → georeferenced 3D pipeline.

- Vite + React + TS, Tailwind v4, react-router, @tanstack/react-query
- `npm run dev` — dev server, proxies `/api` → `http://localhost:8000`
- `npm run build` — typecheck + production build
- `npm run lint` — oxlint

## Mocks

The backend API does not exist yet. When `VITE_USE_MOCKS === "1"` (default in
`.env.development`) all data comes from an in-memory mock in `src/api/mock.ts`:

- new jobs advance one stage every 1.5 s with a ~25% chance of a warning
- seeded `real-demo` (done, real pipeline numbers) and `bad-telemetry` (failed at georef)

Set `VITE_USE_MOCKS=0` to hit the real `/api/*` contract in `src/api/client.ts`.

## Routes

`/` jobs table (poll 3 s) · `/new` upload form · `/jobs/:id` stepper + log + results tabs

3D / Measure / Map tab bodies are placeholders (PR 12 / 13). Metrics tab is live.

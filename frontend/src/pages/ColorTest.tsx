/**
 * ColorTest — dev harness for Phase-A colour correctness validation.
 *
 * Access: http://localhost:5173/colortest
 *
 * Renders three WebGL quads via react-three-fiber and runs gl.readPixels
 * to verify that:
 *   1. Vertex-coloured 128-grey quad reads back 128 ± 3 per channel
 *   2. Texture-mapped 128-grey quad reads back 128 ± 3 per channel
 *   3. Pure-red vertex-coloured quad reads back R≈255, G≈0, B≈0
 *
 * Results are displayed in a table and in the browser console.
 * Green = pass, red = fail.
 */

import { Canvas, useThree } from '@react-three/fiber'
import { useEffect, useState } from 'react'
import * as THREE from 'three'
import { srgbToLinear } from '../lib/colormaps'

// -------------------------------------------------------------------------- //
// readback helper
// -------------------------------------------------------------------------- //
function readCentrePixel(
  renderer: THREE.WebGLRenderer,
  width: number,
  height: number,
): [number, number, number] {
  const buf = new Uint8Array(4)
  const gl = renderer.getContext()
  gl.readPixels(
    Math.floor(width / 2),
    Math.floor(height / 2),
    1, 1,
    gl.RGBA,
    gl.UNSIGNED_BYTE,
    buf,
  )
  return [buf[0], buf[1], buf[2]]
}

// -------------------------------------------------------------------------- //
// inner scene — renders quads and fires callbacks
// -------------------------------------------------------------------------- //
interface QuadSceneProps {
  onResults: (results: TestResult[]) => void
}

interface TestResult {
  label: string
  pixel: [number, number, number]
  expected: string
  pass: boolean
}

function QuadScene({ onResults, runId }: QuadSceneProps & { runId: number }) {
  const { gl, size, scene, camera } = useThree()

  useEffect(() => {
    // Give React a frame to mount and render
    const raf = requestAnimationFrame(() => {
      // Force a single render of the current scene
      gl.render(scene, camera)

      const results: TestResult[] = []

      // --- 1. Grey vertex-coloured quad ---
      const greyVertScene = new THREE.Scene()
      const greyVertGeo = new THREE.PlaneGeometry(2, 2)
      const n = greyVertGeo.attributes.position.count
      const greyArr = new Float32Array(n * 3)
      const v = srgbToLinear(128 / 255) // sRGB 128 converted to linear for Three.js sRGB output
      for (let i = 0; i < n * 3; i++) greyArr[i] = v
      greyVertGeo.setAttribute('color', new THREE.BufferAttribute(greyArr, 3))
      const greyVertMat = new THREE.MeshBasicMaterial({
        vertexColors: true,
        toneMapped: false,
      })
      greyVertScene.add(new THREE.Mesh(greyVertGeo, greyVertMat))

      const ortho = new THREE.OrthographicCamera(-1, 1, 1, -1, 0, 1)
      const w = size.width || gl.domElement.width || 256
      const h = size.height || gl.domElement.height || 256

      gl.render(greyVertScene, ortho)
      const p1 = readCentrePixel(gl, w, h)
      const pass1 = p1.every((c) => Math.abs(c - 128) <= 3)
      results.push({
        label: '128-grey vertex-coloured quad',
        pixel: p1,
        expected: '[128±3, 128±3, 128±3]',
        pass: pass1,
      })

      // --- 2. Grey textured quad ---
      const greyTexData = new Uint8Array([128, 128, 128, 255])
      const greyTex = new THREE.DataTexture(greyTexData, 1, 1, THREE.RGBAFormat)
      greyTex.colorSpace = THREE.SRGBColorSpace
      greyTex.needsUpdate = true
      const greyTexScene = new THREE.Scene()
      const greyTexMat = new THREE.MeshBasicMaterial({ map: greyTex, toneMapped: false })
      greyTexScene.add(new THREE.Mesh(new THREE.PlaneGeometry(2, 2), greyTexMat))
      gl.render(greyTexScene, ortho)
      const p2 = readCentrePixel(gl, w, h)
      const pass2 = p2.every((c) => Math.abs(c - 128) <= 3)
      results.push({
        label: '128-grey textured quad',
        pixel: p2,
        expected: '[128±3, 128±3, 128±3]',
        pass: pass2,
      })

      // --- 3. Pure red vertex-coloured quad ---
      const redScene = new THREE.Scene()
      const redGeo = new THREE.PlaneGeometry(2, 2)
      const nRed = redGeo.attributes.position.count
      const redArr = new Float32Array(nRed * 3)
      for (let i = 0; i < nRed; i++) {
        redArr[i * 3 + 0] = 1.0 // R
        redArr[i * 3 + 1] = 0.0 // G
        redArr[i * 3 + 2] = 0.0 // B
      }
      redGeo.setAttribute('color', new THREE.BufferAttribute(redArr, 3))
      const redMat = new THREE.MeshBasicMaterial({ vertexColors: true, toneMapped: false })
      redScene.add(new THREE.Mesh(redGeo, redMat))
      gl.render(redScene, ortho)
      const p3 = readCentrePixel(gl, w, h)
      const pass3 = p3[0] > 240 && p3[1] < 15 && p3[2] < 15
      results.push({
        label: 'Pure-red vertex-coloured quad',
        pixel: p3,
        expected: '[255, 0, 0]',
        pass: pass3,
      })

      console.log('[ColorTest] results', results)
      onResults(results)
    })
    return () => cancelAnimationFrame(raf)
  }, [gl, size, scene, camera, onResults, runId])

  // Render nothing visible; all tests run off-screen
  return null
}

// -------------------------------------------------------------------------- //
// page component
// -------------------------------------------------------------------------- //
export function ColorTest() {
  const [results, setResults] = useState<TestResult[] | null>(null)
  const [runId, setRunId] = useState(0)
  const allPass = results?.every((r) => r.pass) ?? false

  return (
    <div className="min-h-screen bg-slate-950 p-8 font-mono text-slate-200">
      <div className="mb-6 flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-white">Colour Pipeline Test Harness</h1>
          <p className="mt-1 text-sm text-slate-400">
            Phase A — verifies that sRGB colours pass through the WebGL pipeline without
            double-conversion. All pixel reads use <code>gl.readPixels</code>.
          </p>
        </div>
        <button
          type="button"
          onClick={() => {
            setResults(null)
            setRunId((id) => id + 1)
          }}
          className="rounded bg-slate-800 px-3 py-1.5 text-xs text-slate-300 hover:bg-slate-700"
        >
          Rerun Tests
        </button>
      </div>

      {/* Hidden canvas for test rendering */}
      <Canvas
        key={runId}
        style={{
          width: 256,
          height: 256,
          position: 'fixed',
          top: -9999,
          left: -9999,
          pointerEvents: 'none',
        }}
        gl={{
          antialias: false,
          preserveDrawingBuffer: true,
          powerPreference: 'high-performance',
        }}
        onCreated={(state) => {
          state.gl.toneMapping = THREE.NoToneMapping
          state.gl.outputColorSpace = THREE.SRGBColorSpace
        }}
      >
        <QuadScene onResults={setResults} runId={runId} />
      </Canvas>

      {results === null ? (
        <p className="text-slate-500">Running tests…</p>
      ) : (
        <>
          <div
            className={`mb-6 rounded border px-4 py-3 text-sm font-semibold ${
              allPass
                ? 'border-emerald-500/40 bg-emerald-500/10 text-emerald-300'
                : 'border-red-500/40 bg-red-500/10 text-red-300'
            }`}
          >
            {allPass ? '✅ All tests passed' : '❌ One or more tests FAILED'}
          </div>

          <table className="w-full border-collapse text-sm">
            <thead>
              <tr className="border-b border-slate-800 text-xs uppercase text-slate-500">
                <th className="py-2 text-left">Test</th>
                <th className="py-2 text-left">Expected</th>
                <th className="py-2 text-left">Actual pixel</th>
                <th className="py-2 text-left">Result</th>
              </tr>
            </thead>
            <tbody>
              {results.map((r) => (
                <tr key={r.label} className="border-b border-slate-800">
                  <td className="py-3 pr-4 text-slate-300">{r.label}</td>
                  <td className="py-3 pr-4 text-slate-500">{r.expected}</td>
                  <td className="py-3 pr-4 font-mono text-slate-300">
                    [{r.pixel.join(', ')}]
                  </td>
                  <td className={`py-3 font-bold ${r.pass ? 'text-emerald-400' : 'text-red-400'}`}>
                    {r.pass ? '✅ PASS' : '❌ FAIL'}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>

          <p className="mt-6 text-xs text-slate-600">
            Renderer: <code>toneMapping=NoToneMapping</code>,{' '}
            <code>outputColorSpace=SRGBColorSpace</code>.
            Results also logged to browser console.
          </p>
        </>
      )}
    </div>
  )
}

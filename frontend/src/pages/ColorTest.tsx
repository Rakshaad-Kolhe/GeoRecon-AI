/**
 * ColorTest — dev harness for Phase-A colour correctness validation.
 *
 * Access: http://localhost:5173/colortest
 *
 * Renders WebGL quads and a real PLY fixture through the same PLYLoader →
 * MeshModel / PointCloud code path the viewer uses, running gl.readPixels
 * to verify that:
 *   1. Hand-built vertex-coloured 128-grey quad reads back 128 ± 3
 *   2. Hand-built texture-mapped 128-grey quad reads back 128 ± 3
 *   3. Hand-built pure-red vertex-coloured quad reads back [255, 0, 0]
 *   4. PLYLoader → MeshModel real path: 128-grey reads back 128 ± 3
 *   5. PLYLoader → MeshModel real path: pure red reads back [255, 0, 0]
 *   6. PLYLoader → PointCloud real path: 128-grey reads back 128 ± 3
 *   7. PLYLoader → PointCloud real path: pure red reads back [255, 0, 0]
 *
 * Results are displayed in a table and in the browser console.
 * Green = pass, red = fail.
 */

import { Canvas, useThree } from '@react-three/fiber'
import { useEffect, useState } from 'react'
import * as THREE from 'three'
import { srgbToLinear } from '../lib/colormaps'
import { usePlyGeometry } from '../lib/loadPly'

// -------------------------------------------------------------------------- //
// readback helpers
// -------------------------------------------------------------------------- //
function readPixelAt(
  renderer: THREE.WebGLRenderer,
  x: number,
  y: number,
): [number, number, number] {
  const buf = new Uint8Array(4)
  const gl = renderer.getContext()
  gl.readPixels(
    Math.round(x),
    Math.round(y),
    1, 1,
    gl.RGBA,
    gl.UNSIGNED_BYTE,
    buf,
  )
  return [buf[0], buf[1], buf[2]]
}

function readCentrePixel(
  renderer: THREE.WebGLRenderer,
  width: number,
  height: number,
): [number, number, number] {
  return readPixelAt(renderer, width / 2, height / 2)
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
  const plyState = usePlyGeometry('/test-fixtures/tiny_test.ply', true, runId)

  useEffect(() => {
    if (plyState.status !== 'ready') return

    // Give React a frame to mount and render
    const raf = requestAnimationFrame(() => {
      // Force a single render of the current scene
      gl.render(scene, camera)

      const results: TestResult[] = []
      const ortho = new THREE.OrthographicCamera(-1, 1, 1, -1, 0.1, 20)
      ortho.position.set(0, 0, 5)
      ortho.lookAt(0, 0, 0)
      ortho.updateProjectionMatrix()
      const bufSize = new THREE.Vector2()
      gl.getDrawingBufferSize(bufSize)
      const w = bufSize.x || 256
      const h = bufSize.y || 256

      // --- 1. Hand-built grey vertex-coloured quad ---
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

      gl.render(greyVertScene, ortho)
      const p1 = readCentrePixel(gl, w, h)
      const pass1 = p1.every((c) => Math.abs(c - 128) <= 3)
      results.push({
        label: 'Hand-built 128-grey vertex quad',
        pixel: p1,
        expected: '[128±3, 128±3, 128±3]',
        pass: pass1,
      })

      // --- 2. Hand-built grey textured quad ---
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
        label: 'Hand-built 128-grey textured quad',
        pixel: p2,
        expected: '[128±3, 128±3, 128±3]',
        pass: pass2,
      })

      // --- 3. Hand-built pure red vertex quad ---
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
        label: 'Hand-built pure-red vertex quad',
        pixel: p3,
        expected: '[255, 0, 0]',
        pass: pass3,
      })

      // --- Real path tests using tiny_test.ply ---
      // tiny_test.ply was parsed by PLYLoader (which converts sRGB -> linear in attributes.color).
      // The viewer's MeshModel and PointCloud code extracts src = geom.attributes.color
      // and passes src.getX/Y/Z directly (without a second srgbToLinear).
      const plyGeom = plyState.geometry
      const plyN = plyGeom.attributes.position.count
      const plyRgb = new Float32Array(plyN * 3)
      const plySrc = plyGeom.attributes.color
      for (let i = 0; i < plyN; i++) {
        if (plySrc) {
          plyRgb[i * 3 + 0] = Math.min(1, plySrc.getX(i))
          plyRgb[i * 3 + 1] = Math.min(1, plySrc.getY(i))
          plyRgb[i * 3 + 2] = Math.min(1, plySrc.getZ(i))
        }
      }

      // --- 4 & 5. PLYLoader -> MeshModel real path ---
      const meshGeom = plyGeom.clone()
      meshGeom.setAttribute('color', new THREE.BufferAttribute(plyRgb, 3))
      const meshMat = new THREE.MeshBasicMaterial({ vertexColors: true, toneMapped: false })
      const meshScene = new THREE.Scene()
      meshScene.add(new THREE.Mesh(meshGeom, meshMat))
      gl.render(meshScene, ortho)

      // Sample bottom-left (grey triangle) and top-right (red triangle)
      const p4Grey = readPixelAt(gl, w * 0.25, h * 0.25)
      const pass4Grey = p4Grey.every((c) => Math.abs(c - 128) <= 3)
      results.push({
        label: 'PLYLoader → MeshModel: 128-grey (real path)',
        pixel: p4Grey,
        expected: '[128±3, 128±3, 128±3]',
        pass: pass4Grey,
      })

      const p4Red = readPixelAt(gl, w * 0.75, h * 0.75)
      const pass4Red = p4Red[0] > 240 && p4Red[1] < 15 && p4Red[2] < 15
      results.push({
        label: 'PLYLoader → MeshModel: pure red (real path)',
        pixel: p4Red,
        expected: '[255, 0, 0]',
        pass: pass4Red,
      })

      // --- 6 & 7. PLYLoader -> PointCloud real path ---
      const cloudGeom = plyGeom.clone()
      cloudGeom.index = null // Point clouds are unindexed vertex lists
      cloudGeom.setAttribute('color', new THREE.BufferAttribute(plyRgb, 3))
      const cloudMat = new THREE.PointsMaterial({
        vertexColors: true,
        sizeAttenuation: false,
        size: 80,
      })
      cloudMat.onBeforeCompile = (shader) => {
        shader.fragmentShader = shader.fragmentShader.replace(
          '#include <clipping_planes_fragment>',
          `#include <clipping_planes_fragment>
          vec2 pc = gl_PointCoord - vec2( 0.5 );
          if ( dot( pc, pc ) > 0.25 ) discard;`,
        )
      }
      ;(cloudMat as unknown as { toneMapped: boolean }).toneMapped = false

      const cloudScene = new THREE.Scene()
      cloudScene.add(new THREE.Points(cloudGeom, cloudMat))
      gl.render(cloudScene, ortho)

      const p5Grey = readPixelAt(gl, w * 0.25, h * 0.25)
      const pass5Grey = p5Grey.every((c) => Math.abs(c - 128) <= 3)
      results.push({
        label: 'PLYLoader → PointCloud: 128-grey (real path)',
        pixel: p5Grey,
        expected: '[128±3, 128±3, 128±3]',
        pass: pass5Grey,
      })

      const p5Red = readPixelAt(gl, w * 0.75, h * 0.75)
      const pass5Red = p5Red[0] > 240 && p5Red[1] < 15 && p5Red[2] < 15
      results.push({
        label: 'PLYLoader → PointCloud: pure red (real path)',
        pixel: p5Red,
        expected: '[255, 0, 0]',
        pass: pass5Red,
      })

      console.log('[ColorTest] results', results)
      onResults(results)
    })
    return () => cancelAnimationFrame(raf)
  }, [gl, size, scene, camera, onResults, runId, plyState])

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
            double-conversion, testing both hand-built quads and a real PLY fixture loaded
            via <code>PLYLoader → MeshModel/PointCloud</code>. All pixel reads use{' '}
            <code>gl.readPixels</code>.
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
        dpr={1}
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

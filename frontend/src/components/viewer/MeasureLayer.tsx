import { Html, Line } from '@react-three/drei'
import { useMemo } from 'react'
import * as THREE from 'three'
import {
  centroidXY,
  midpoint,
  type MeasureTool,
  type Measurement,
  type Vec3,
} from '../../lib/measure'

interface Props {
  measurements: Measurement[]
  draft: { tool: MeasureTool; points: Vec3[] } | null
  hover: Vec3 | null
  markerRadius: number
}

const CYAN = '#22d3ee'

function Marker({ p, r }: { p: Vec3; r: number }) {
  return (
    <mesh position={[p.x, p.y, p.z]}>
      <sphereGeometry args={[r, 12, 12]} />
      <meshBasicMaterial color={CYAN} depthTest={false} />
    </mesh>
  )
}

function Label({ p, text }: { p: Vec3; text: string }) {
  return (
    <Html position={[p.x, p.y, p.z]} center distanceFactor={220} zIndexRange={[10, 0]}>
      <div className="pointer-events-none whitespace-nowrap rounded border border-cyan-400/40 bg-slate-950/90 px-1.5 py-0.5 font-mono text-[10px] text-cyan-200">
        {text}
      </div>
    </Html>
  )
}

function AreaFill({ pts }: { pts: Vec3[] }) {
  const shape = useMemo(() => {
    const s = new THREE.Shape()
    pts.forEach((p, i) => (i === 0 ? s.moveTo(p.x, p.y) : s.lineTo(p.x, p.y)))
    s.closePath()
    return s
  }, [pts])
  return (
    <mesh position={[0, 0, centroidXY(pts).z]}>
      <shapeGeometry args={[shape]} />
      <meshBasicMaterial
        color={CYAN}
        transparent
        opacity={0.15}
        side={THREE.DoubleSide}
        depthWrite={false}
      />
    </mesh>
  )
}

function MeasureShape({
  tool,
  points,
  label,
  closed,
  r,
}: {
  tool: MeasureTool
  points: Vec3[]
  label?: string
  closed: boolean
  r: number
}) {
  const line: [number, number, number][] = points.map((p) => [p.x, p.y, p.z])
  if (closed && points.length >= 3) line.push([points[0].x, points[0].y, points[0].z])

  let labelPos: Vec3 | null = null
  if (label) {
    if (tool === 'area') labelPos = centroidXY(points)
    else if (points.length >= 2) labelPos = midpoint(points[0], points[1])
    else labelPos = points[0] ?? null
  }

  return (
    <group>
      {tool === 'area' && closed && points.length >= 3 && <AreaFill pts={points} />}
      {line.length >= 2 && (
        <Line
          points={line}
          color={CYAN}
          lineWidth={2}
          dashed={!closed}
          dashSize={r}
          gapSize={r}
        />
      )}
      {points.map((p, i) => (
        <Marker key={i} p={p} r={r} />
      ))}
      {label && labelPos && <Label p={labelPos} text={label} />}
    </group>
  )
}

export function MeasureLayer({ measurements, draft, hover, markerRadius }: Props) {
  return (
    <group>
      {measurements.map((mm) => (
        <MeasureShape
          key={mm.id}
          tool={mm.tool}
          points={mm.points}
          label={mm.label}
          closed={mm.tool === 'area'}
          r={markerRadius}
        />
      ))}

      {draft && draft.points.length > 0 && (
        <MeasureShape
          tool={draft.tool}
          points={draft.points}
          label={
            draft.tool === 'area'
              ? `${draft.points.length} pt${draft.points.length > 1 ? 's' : ''}`
              : undefined
          }
          closed={false}
          r={markerRadius}
        />
      )}

      {hover && (
        <mesh position={[hover.x, hover.y, hover.z]}>
          <ringGeometry args={[markerRadius * 1.6, markerRadius * 2.4, 20]} />
          <meshBasicMaterial
            color={CYAN}
            transparent
            opacity={0.9}
            side={THREE.DoubleSide}
            depthTest={false}
          />
        </mesh>
      )}
    </group>
  )
}

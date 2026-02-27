import clsx from 'clsx'

interface ScoreCardProps {
  score: number // 0.0 to 1.0
  compact?: boolean
}

function getGrade(score: number): { letter: string; color: string; bgColor: string } {
  const pct = score * 100
  if (pct >= 90) return { letter: 'A', color: 'text-green-700', bgColor: 'bg-green-100' }
  if (pct >= 80) return { letter: 'B', color: 'text-lime-700', bgColor: 'bg-lime-100' }
  if (pct >= 70) return { letter: 'C', color: 'text-yellow-700', bgColor: 'bg-yellow-100' }
  if (pct >= 60) return { letter: 'D', color: 'text-orange-700', bgColor: 'bg-orange-100' }
  return { letter: 'F', color: 'text-red-700', bgColor: 'bg-red-100' }
}

export default function ScoreCard({ score, compact = false }: ScoreCardProps) {
  const grade = getGrade(score)
  const pct = (score * 100).toFixed(0)

  if (compact) {
    return (
      <span className={clsx('inline-flex items-center gap-1 px-2 py-1 rounded font-medium', grade.color, grade.bgColor)}>
        <span className="text-lg font-bold">{grade.letter}</span>
        <span className="text-sm">{pct}%</span>
      </span>
    )
  }

  return (
    <div className="flex items-center gap-4">
      <div className={clsx('w-16 h-16 rounded-lg flex items-center justify-center', grade.bgColor)}>
        <span className={clsx('text-3xl font-bold', grade.color)}>{grade.letter}</span>
      </div>
      <div>
        <div className="text-3xl font-bold text-gray-900">{pct}%</div>
        <div className="text-sm text-gray-600">Quality Score</div>
      </div>
    </div>
  )
}

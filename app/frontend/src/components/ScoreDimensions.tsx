import { useState } from 'react'
import {
  CheckCircle2,
  XCircle,
  AlertTriangle,
  Info,
  ChevronDown,
  ChevronUp,
  Shield,
  Zap,
  Settings,
  FileCode,
  BookOpen,
} from 'lucide-react'

interface DimensionCheck {
  name: string
  passed: boolean
  message: string
  severity: string
  expected?: string
  actual?: string
}

interface ScoreDimension {
  name: string
  score: number
  weight: number
  description: string
  passed_count: number
  failed_count: number
  grade: string
  checks?: DimensionCheck[]
}

interface ScoreDimensionsProps {
  dimensions: ScoreDimension[]
}

const DIMENSION_ICONS: Record<string, React.ReactNode> = {
  code_quality: <Shield size={18} />,
  standards: <BookOpen size={18} />,
  performance: <Zap size={18} />,
  parameterization: <Settings size={18} />,
  verbosity: <FileCode size={18} />,
  data_accuracy: <CheckCircle2 size={18} />,
}

const DIMENSION_LABELS: Record<string, string> = {
  code_quality: 'Code Quality',
  standards: 'Standards',
  performance: 'Performance',
  parameterization: 'Parameterization',
  verbosity: 'Verbosity',
  data_accuracy: 'Data Accuracy',
}

function getScoreColor(score: number) {
  if (score >= 80) return 'text-green-600'
  if (score >= 60) return 'text-yellow-600'
  return 'text-red-600'
}

function getScoreBg(score: number) {
  if (score >= 80) return 'bg-green-500'
  if (score >= 60) return 'bg-yellow-500'
  return 'bg-red-500'
}

function getGradeBg(grade: string) {
  if (grade === 'A') return 'bg-green-100 text-green-800'
  if (grade === 'B') return 'bg-blue-100 text-blue-800'
  if (grade === 'C') return 'bg-yellow-100 text-yellow-800'
  if (grade === 'D') return 'bg-orange-100 text-orange-800'
  return 'bg-red-100 text-red-800'
}

function getSeverityIcon(severity: string, passed: boolean) {
  if (passed) return <CheckCircle2 className="text-green-600 flex-shrink-0" size={16} />
  if (severity === 'error') return <XCircle className="text-red-600 flex-shrink-0" size={16} />
  if (severity === 'warning') return <AlertTriangle className="text-yellow-600 flex-shrink-0" size={16} />
  return <Info className="text-blue-600 flex-shrink-0" size={16} />
}

export default function ScoreDimensions({ dimensions }: ScoreDimensionsProps) {
  const [expandedDimension, setExpandedDimension] = useState<string | null>(null)

  if (!dimensions || dimensions.length === 0) {
    return (
      <div className="bg-white rounded-xl shadow-sm border p-6 text-center text-gray-500">
        <Info size={24} className="mx-auto mb-2" />
        No dimension scores available. Run validation to generate scores.
      </div>
    )
  }

  const toggleDimension = (name: string) => {
    setExpandedDimension(expandedDimension === name ? null : name)
  }

  return (
    <div className="space-y-4">
      {/* Score bar chart */}
      <div className="bg-white rounded-xl shadow-sm border p-6">
        <h3 className="text-sm font-semibold text-gray-700 mb-4 uppercase tracking-wide">
          Quality Dimensions
        </h3>
        <div className="space-y-3">
          {dimensions.map((dim) => (
            <div key={dim.name} className="flex items-center gap-3">
              <div className="w-32 flex items-center gap-2 text-sm text-gray-700">
                {DIMENSION_ICONS[dim.name] || <Shield size={18} />}
                <span className="truncate">{DIMENSION_LABELS[dim.name] || dim.name}</span>
              </div>
              <div className="flex-1 h-5 bg-gray-100 rounded-full overflow-hidden">
                <div
                  className={`h-full rounded-full transition-all duration-500 ${getScoreBg(dim.score)}`}
                  style={{ width: `${dim.score}%` }}
                />
              </div>
              <div className={`w-12 text-right text-sm font-semibold ${getScoreColor(dim.score)}`}>
                {dim.score}%
              </div>
              <span className={`text-xs font-bold px-2 py-0.5 rounded ${getGradeBg(dim.grade)}`}>
                {dim.grade}
              </span>
            </div>
          ))}
        </div>
      </div>

      {/* Expandable dimension details */}
      <div className="bg-white rounded-xl shadow-sm border divide-y">
        {dimensions.map((dim) => (
          <div key={dim.name}>
            <button
              onClick={() => toggleDimension(dim.name)}
              className="w-full p-4 flex items-center justify-between hover:bg-gray-50 transition-colors"
            >
              <div className="flex items-center gap-3">
                {DIMENSION_ICONS[dim.name] || <Shield size={18} />}
                <div className="text-left">
                  <div className="font-medium text-gray-900">
                    {DIMENSION_LABELS[dim.name] || dim.name}
                  </div>
                  <div className="text-xs text-gray-500">{dim.description}</div>
                </div>
              </div>
              <div className="flex items-center gap-3">
                <div className="text-sm text-gray-500">
                  <span className="text-green-600 font-medium">{dim.passed_count}</span>
                  {' / '}
                  {dim.passed_count + dim.failed_count} checks
                </div>
                {expandedDimension === dim.name ? (
                  <ChevronUp size={18} className="text-gray-400" />
                ) : (
                  <ChevronDown size={18} className="text-gray-400" />
                )}
              </div>
            </button>

            {expandedDimension === dim.name && dim.checks && (
              <div className="px-4 pb-4 space-y-2">
                {dim.checks.map((check, idx) => (
                  <div
                    key={idx}
                    className={`p-3 rounded-lg ${
                      check.passed ? 'bg-green-50' :
                      check.severity === 'error' ? 'bg-red-50' :
                      check.severity === 'warning' ? 'bg-yellow-50' : 'bg-blue-50'
                    }`}
                  >
                    <div className="flex items-start gap-2">
                      {getSeverityIcon(check.severity, check.passed)}
                      <div className="flex-1 min-w-0">
                        <div className="font-medium text-sm text-gray-900">
                          {check.name.replace(/_/g, ' ')}
                        </div>
                        <div className="text-sm text-gray-600">{check.message}</div>
                        {check.expected && !check.passed && (
                          <div className="text-xs text-gray-500 mt-1">
                            Expected: {check.expected}
                            {check.actual && ` | Actual: ${check.actual}`}
                          </div>
                        )}
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>
        ))}
      </div>
    </div>
  )
}

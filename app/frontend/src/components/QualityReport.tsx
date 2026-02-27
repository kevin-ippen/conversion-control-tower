import { CheckCircle2, XCircle, AlertTriangle, Info, TrendingUp } from 'lucide-react'
import ScoreDimensions from './ScoreDimensions'

interface QualityCheck {
  check_id: string
  check_name: string
  category: string
  passed: boolean
  severity: string
  message: string
  details?: string
  suggestion?: string
}

interface DimensionData {
  name: string
  score: number
  weight: number
  description: string
  passed_count: number
  failed_count: number
  grade: string
  checks?: Array<{
    name: string
    passed: boolean
    message: string
    severity: string
    expected?: string
    actual?: string
  }>
}

interface QualityReportData {
  overall_score: number
  checks: QualityCheck[]
  summary: string
  recommendations: string[]
  dimensions?: DimensionData[]
}

interface QualityReportProps {
  report: QualityReportData | null
}

export default function QualityReport({ report }: QualityReportProps) {
  if (!report) {
    return (
      <div className="bg-white rounded-xl shadow-sm border p-6 text-center text-gray-500">
        <Info size={24} className="mx-auto mb-2" />
        Quality report not available yet
      </div>
    )
  }

  const getScoreColor = (score: number) => {
    if (score >= 80) return 'text-green-600'
    if (score >= 60) return 'text-yellow-600'
    return 'text-red-600'
  }

  const getScoreBg = (score: number) => {
    if (score >= 80) return 'bg-green-100'
    if (score >= 60) return 'bg-yellow-100'
    return 'bg-red-100'
  }

  const getSeverityIcon = (severity: string, passed: boolean) => {
    if (passed) return <CheckCircle2 className="text-green-600" size={18} />
    if (severity === 'error') return <XCircle className="text-red-600" size={18} />
    if (severity === 'warning') return <AlertTriangle className="text-yellow-600" size={18} />
    return <Info className="text-blue-600" size={18} />
  }

  const groupedChecks = report.checks.reduce((acc, check) => {
    if (!acc[check.category]) acc[check.category] = []
    acc[check.category].push(check)
    return acc
  }, {} as Record<string, QualityCheck[]>)

  const passedCount = report.checks.filter(c => c.passed).length
  const totalCount = report.checks.length

  return (
    <div className="bg-white rounded-xl shadow-sm border">
      <div className="p-6 border-b">
        <div className="flex items-center justify-between">
          <div>
            <h2 className="text-lg font-semibold flex items-center gap-2">
              <TrendingUp size={20} />
              Quality Report
            </h2>
            <p className="text-sm text-gray-500 mt-1">{report.summary}</p>
          </div>
          <div className={`text-center p-4 rounded-xl ${getScoreBg(report.overall_score <= 1 ? report.overall_score * 100 : report.overall_score)}`}>
            <div className={`text-3xl font-bold ${getScoreColor(report.overall_score <= 1 ? report.overall_score * 100 : report.overall_score)}`}>
              {report.overall_score <= 1 ? Math.round(report.overall_score * 100) : Math.round(report.overall_score)}
            </div>
            <div className="text-xs text-gray-600">Quality Score</div>
          </div>
        </div>

        {/* Pass rate bar */}
        <div className="mt-4">
          <div className="flex justify-between text-sm text-gray-600 mb-1">
            <span>Checks Passed</span>
            <span>{passedCount} / {totalCount}</span>
          </div>
          <div className="h-2 bg-gray-200 rounded-full overflow-hidden">
            <div
              className="h-full bg-green-500"
              style={{ width: `${(passedCount / totalCount) * 100}%` }}
            />
          </div>
        </div>
      </div>

      {/* Multi-dimensional scores */}
      {report.dimensions && report.dimensions.length > 0 && (
        <div className="p-4">
          <ScoreDimensions dimensions={report.dimensions} />
        </div>
      )}

      {/* Checks by category (legacy fallback) */}
      <div className="divide-y">
        {Object.entries(groupedChecks).map(([category, checks]) => (
          <div key={category} className="p-4">
            <h3 className="font-medium text-gray-700 mb-3 uppercase text-xs tracking-wide">
              {category}
            </h3>
            <div className="space-y-2">
              {checks.map((check) => (
                <div
                  key={check.check_id}
                  className={`p-3 rounded-lg ${
                    check.passed ? 'bg-green-50' :
                    check.severity === 'error' ? 'bg-red-50' :
                    check.severity === 'warning' ? 'bg-yellow-50' : 'bg-blue-50'
                  }`}
                >
                  <div className="flex items-start gap-2">
                    {getSeverityIcon(check.severity, check.passed)}
                    <div className="flex-1">
                      <div className="font-medium text-gray-900">{check.check_name}</div>
                      <div className="text-sm text-gray-600">{check.message}</div>
                      {check.suggestion && (
                        <div className="text-sm text-blue-700 mt-1">
                          Suggestion: {check.suggestion}
                        </div>
                      )}
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </div>
        ))}
      </div>

      {/* Recommendations */}
      {report.recommendations.length > 0 && (
        <div className="p-4 bg-blue-50 border-t">
          <h3 className="font-medium text-blue-900 mb-2">Recommendations</h3>
          <ul className="space-y-1">
            {report.recommendations.map((rec, i) => (
              <li key={i} className="text-sm text-blue-800 flex items-start gap-2">
                <span className="text-blue-500">•</span>
                {rec}
              </li>
            ))}
          </ul>
        </div>
      )}
    </div>
  )
}

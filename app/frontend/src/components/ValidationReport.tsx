import { CheckCircle2, XCircle, AlertTriangle, Info } from 'lucide-react'
import clsx from 'clsx'

interface ValidationResult {
  validation_id: string
  check_name: string
  passed: boolean
  expected: string | null
  actual: string | null
  message: string
  severity: string
  category: string
}

interface ValidationReportProps {
  results: ValidationResult[]
}

const severityIcons: Record<string, React.ReactNode> = {
  error: <XCircle size={18} />,
  warning: <AlertTriangle size={18} />,
  info: <Info size={18} />,
}

const categoryColors: Record<string, string> = {
  completeness: 'bg-blue-100 text-blue-700',
  accuracy: 'bg-purple-100 text-purple-700',
  logic: 'bg-amber-100 text-amber-700',
  error_handling: 'bg-teal-100 text-teal-700',
}

export default function ValidationReport({ results }: ValidationReportProps) {
  const passed = results.filter((r) => r.passed)
  const failed = results.filter((r) => !r.passed)

  return (
    <div className="space-y-6">
      {/* Summary */}
      <div className="flex gap-4">
        <div className="flex-1 bg-green-50 rounded-lg p-4 text-center">
          <div className="text-2xl font-bold text-green-700">{passed.length}</div>
          <div className="text-sm text-green-600">Passed</div>
        </div>
        <div className="flex-1 bg-red-50 rounded-lg p-4 text-center">
          <div className="text-2xl font-bold text-red-700">{failed.length}</div>
          <div className="text-sm text-red-600">Failed</div>
        </div>
      </div>

      {/* Failed Checks */}
      {failed.length > 0 && (
        <div>
          <h4 className="font-medium text-gray-900 mb-2">Failed Checks</h4>
          <div className="space-y-2">
            {failed.map((result) => (
              <ValidationItem key={result.validation_id} result={result} />
            ))}
          </div>
        </div>
      )}

      {/* Passed Checks */}
      {passed.length > 0 && (
        <div>
          <h4 className="font-medium text-gray-900 mb-2">Passed Checks</h4>
          <div className="space-y-2">
            {passed.map((result) => (
              <ValidationItem key={result.validation_id} result={result} />
            ))}
          </div>
        </div>
      )}
    </div>
  )
}

function ValidationItem({ result }: { result: ValidationResult }) {
  return (
    <div
      className={clsx(
        'flex items-start gap-3 p-3 rounded-lg',
        result.passed ? 'bg-green-50' : 'bg-red-50'
      )}
    >
      <span className={result.passed ? 'text-green-600' : 'text-red-600'}>
        {result.passed ? <CheckCircle2 size={18} /> : severityIcons[result.severity]}
      </span>
      <div className="flex-1 min-w-0">
        <div className="flex items-center gap-2 flex-wrap">
          <span className="font-medium text-gray-900">{result.check_name}</span>
          <span className={clsx('px-2 py-0.5 text-xs rounded-full', categoryColors[result.category] || 'bg-gray-100 text-gray-700')}>
            {result.category}
          </span>
        </div>
        <p className="text-sm text-gray-600 mt-1">{result.message}</p>
        {result.expected && (
          <div className="text-xs text-gray-500 mt-1">
            Expected: {result.expected} | Actual: {result.actual}
          </div>
        )}
      </div>
    </div>
  )
}

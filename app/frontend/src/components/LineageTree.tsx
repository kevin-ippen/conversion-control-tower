import { Link } from 'react-router-dom'
import { GitBranch, ArrowRight, CheckCircle2, XCircle, Clock, Loader2 } from 'lucide-react'

interface ConversionJob {
  job_id: string
  job_name: string
  status: string
  quality_score: number | null
  created_at: string
  ai_model?: string
}

interface LineageTreeProps {
  original: ConversionJob | null
  refinements: ConversionJob[]
  currentJobId: string
  isLoading?: boolean
}

function StatusIcon({ status }: { status: string }) {
  switch (status) {
    case 'completed':
      return <CheckCircle2 className="text-green-500" size={16} />
    case 'failed':
      return <XCircle className="text-red-500" size={16} />
    case 'pending':
      return <Clock className="text-gray-400" size={16} />
    default:
      return <Loader2 className="text-orange-500 animate-spin" size={16} />
  }
}

function ScoreBadge({ score }: { score: number | null }) {
  if (score === null) return <span className="text-gray-400 text-xs">-</span>

  const color = score >= 0.8 ? 'bg-green-100 text-green-800' :
                score >= 0.6 ? 'bg-yellow-100 text-yellow-800' :
                'bg-red-100 text-red-800'

  return (
    <span className={`px-2 py-0.5 rounded-full text-xs font-medium ${color}`}>
      {(score * 100).toFixed(0)}%
    </span>
  )
}

export default function LineageTree({ original, refinements, currentJobId, isLoading }: LineageTreeProps) {
  if (isLoading) {
    return (
      <div className="bg-white rounded-xl shadow-sm border p-6">
        <div className="flex items-center gap-3">
          <Loader2 className="animate-spin text-gray-400" size={20} />
          <span className="text-gray-500">Loading lineage...</span>
        </div>
      </div>
    )
  }

  if (!original && refinements.length === 0) {
    return (
      <div className="bg-white rounded-xl shadow-sm border p-6">
        <div className="flex items-center gap-3 text-gray-500">
          <GitBranch size={20} />
          <span>No refinement history available</span>
        </div>
      </div>
    )
  }

  const allJobs = original ? [original, ...refinements] : refinements

  return (
    <div className="bg-white rounded-xl shadow-sm border p-6">
      <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
        <GitBranch size={20} />
        Conversion Lineage
      </h3>

      <div className="relative">
        {/* Timeline line */}
        {allJobs.length > 1 && (
          <div className="absolute left-5 top-8 bottom-8 w-0.5 bg-gray-200" />
        )}

        <div className="space-y-4">
          {allJobs.map((job, index) => {
            const isCurrent = job.job_id === currentJobId
            const isOriginal = index === 0

            return (
              <div key={job.job_id} className="relative flex items-start gap-4">
                {/* Node */}
                <div className={`relative z-10 w-10 h-10 rounded-full flex items-center justify-center ${
                  isCurrent ? 'bg-orange-100 border-2 border-orange-500' :
                  isOriginal ? 'bg-blue-100 border-2 border-blue-500' :
                  'bg-gray-100 border-2 border-gray-300'
                }`}>
                  <StatusIcon status={job.status} />
                </div>

                {/* Content */}
                <div className={`flex-1 p-4 rounded-lg ${
                  isCurrent ? 'bg-orange-50 border border-orange-200' :
                  'bg-gray-50 border border-gray-200'
                }`}>
                  <div className="flex items-center justify-between mb-2">
                    <div className="flex items-center gap-2">
                      {isCurrent ? (
                        <span className="font-semibold text-gray-900">{job.job_name}</span>
                      ) : (
                        <Link
                          to={`/conversions/${job.job_id}`}
                          className="font-semibold text-blue-600 hover:underline"
                        >
                          {job.job_name}
                        </Link>
                      )}
                      {isOriginal && (
                        <span className="px-2 py-0.5 text-xs bg-blue-100 text-blue-800 rounded-full">
                          Original
                        </span>
                      )}
                      {!isOriginal && (
                        <span className="px-2 py-0.5 text-xs bg-purple-100 text-purple-800 rounded-full">
                          Refinement {index}
                        </span>
                      )}
                      {isCurrent && (
                        <span className="px-2 py-0.5 text-xs bg-orange-100 text-orange-800 rounded-full">
                          Current
                        </span>
                      )}
                    </div>
                    <ScoreBadge score={job.quality_score} />
                  </div>

                  <div className="flex items-center gap-4 text-sm text-gray-600">
                    <span>{new Date(job.created_at).toLocaleString()}</span>
                    {job.ai_model && (
                      <span className="text-gray-400">• {job.ai_model}</span>
                    )}
                  </div>

                  {/* Score change indicator */}
                  {index > 0 && allJobs[index - 1].quality_score !== null && job.quality_score !== null && (
                    <div className="mt-2 flex items-center gap-1 text-xs">
                      <ArrowRight size={12} className="text-gray-400" />
                      {job.quality_score > allJobs[index - 1].quality_score! ? (
                        <span className="text-green-600">
                          +{((job.quality_score - allJobs[index - 1].quality_score!) * 100).toFixed(1)}% improvement
                        </span>
                      ) : job.quality_score < allJobs[index - 1].quality_score! ? (
                        <span className="text-red-600">
                          {((job.quality_score - allJobs[index - 1].quality_score!) * 100).toFixed(1)}% change
                        </span>
                      ) : (
                        <span className="text-gray-500">No score change</span>
                      )}
                    </div>
                  )}
                </div>
              </div>
            )
          })}
        </div>
      </div>

      {allJobs.length === 1 && (
        <p className="mt-4 text-sm text-gray-500 text-center">
          This is the original conversion with no refinements yet.
        </p>
      )}
    </div>
  )
}

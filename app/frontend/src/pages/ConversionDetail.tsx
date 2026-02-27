import { useParams, Link } from 'react-router-dom'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { useState, useEffect } from 'react'
import {
  ArrowLeft,
  Play,
  CheckCircle2,
  XCircle,
  Rocket,
  RefreshCw,
  Loader2,
  FileText,
  Sparkles,
  Wrench,
  AlertCircle,
  GitBranch,
  Code2,
  History,
  GitCompare,
  LayoutDashboard,
  FlaskConical,
} from 'lucide-react'
import axios from 'axios'
import StatusBadge from '../components/StatusBadge'
import ScoreCard from '../components/ScoreCard'
import CodeCompare from '../components/CodeCompare'
import QualityReport from '../components/QualityReport'
import DataCompare from '../components/DataCompare'
import LineageTree from '../components/LineageTree'

type TabType = 'overview' | 'code' | 'history' | 'comparisons'

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

interface QualityReportData {
  overall_score: number
  checks: QualityCheck[]
  summary: string
  recommendations: string[]
}

interface ReferenceFile {
  file_id: string
  file_name: string
  file_type: string
}

interface ConversionJob {
  job_id: string
  job_name: string
  source_type: string
  source_path: string
  output_path: string | null
  status: string
  quality_score: number | null
  created_by: string | null
  created_at: string
  updated_at: string
  completed_at: string | null
  error_message: string | null
  ai_model?: string
  conversion_instructions?: string
  reference_files?: ReferenceFile[]
  quality_report?: QualityReportData
  metadata?: Record<string, string>
}

interface CodeCompareData {
  job_id: string
  source_file: string
  source_content: string
  output_files: { name: string; content: string; type: string }[]
}

interface ColumnSchema {
  name: string
  type: string
}

interface DataSample {
  columns: string[]
  rows: (string | number | null)[][]
}

interface DataSource {
  location: string
  catalog?: string
  schema_name?: string
  table?: string
  row_count: number
  column_count: number
  columns: ColumnSchema[]
  sample: DataSample
}

interface DataCompareData {
  job_id: string
  original: DataSource | null
  converted: DataSource | null
  comparison?: {
    row_count_match: boolean
    schema_match: boolean
    sample_match_rate: number
    mismatched_columns: string[]
    summary: string
  }
}

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

interface StatusEvent {
  progress?: number
  stage?: string
  message?: string
  status?: string
  error?: string
}

interface IssueSummary {
  job_id: string
  summary: string
  issues: { check_name: string; message: string; severity: string; suggestion?: string }[]
  severity: string
  issue_count: number
}

interface RefineResult {
  status: string
  original_job_id: string
  refinement_job_id: string
  databricks_run_id: string
  refinement_prompt: string
  ai_model: string
  message: string
}

interface LineageData {
  original: ConversionJob | null
  refinements: ConversionJob[]
}

export default function ConversionDetail() {
  const { jobId } = useParams<{ jobId: string }>()
  const queryClient = useQueryClient()
  const [activeTab, setActiveTab] = useState<TabType>('overview')
  const [statusEvents, setStatusEvents] = useState<StatusEvent[]>([])
  const [currentProgress, setCurrentProgress] = useState(0)
  const [currentStage, setCurrentStage] = useState('')
  const [runPageUrl, setRunPageUrl] = useState<string | null>(null)
  const [issueSummary, setIssueSummary] = useState<IssueSummary | null>(null)
  const [refinementPrompt, setRefinementPrompt] = useState<string>('')
  const [showRefinementModal, setShowRefinementModal] = useState(false)
  const [refineResult, setRefineResult] = useState<RefineResult | null>(null)

  const { data: job, isLoading } = useQuery<ConversionJob>({
    queryKey: ['conversion', jobId],
    queryFn: async () => {
      const { data } = await axios.get(`/api/conversions/${jobId}`)
      return data
    },
  })

  const { data: validations } = useQuery<ValidationResult[]>({
    queryKey: ['validation', jobId],
    queryFn: async () => {
      try {
        const { data } = await axios.get(`/api/validation/${jobId}/results`)
        return data
      } catch {
        return []
      }
    },
    enabled: !!job && job.status === 'completed',
  })

  // Fetch code compare data for completed jobs
  const { data: codeCompare } = useQuery<CodeCompareData>({
    queryKey: ['codeCompare', jobId],
    queryFn: async () => {
      const { data } = await axios.get(`/api/files/${jobId}/compare`)
      return data
    },
    enabled: !!job && job.status === 'completed',
  })

  // Fetch data output compare for completed jobs
  const { data: dataCompare, refetch: refetchDataCompare, isLoading: isDataCompareLoading } = useQuery<DataCompareData>({
    queryKey: ['dataCompare', jobId],
    queryFn: async () => {
      const { data } = await axios.get(`/api/validation/${jobId}/data-compare`)
      return data
    },
    enabled: !!job && job.status === 'completed',
  })

  // Fetch validation pipeline status for completed jobs
  const { data: pipelineStatus } = useQuery<{
    conversion_job_id: string
    status: string
    steps?: Record<string, { run_id?: string; status: string; run_page_url?: string }>
    message?: string
    error?: string
  }>({
    queryKey: ['pipelineStatus', jobId],
    queryFn: async () => {
      const { data } = await axios.get(`/api/validation/${jobId}/pipeline-status`)
      return data
    },
    enabled: !!job && job.status === 'completed',
    refetchInterval: (query) => {
      const status = query.state.data?.status
      return status === 'running' || status === 'started' ? 5000 : false
    },
  })

  // Auto-refresh data compare when pipeline completes
  useEffect(() => {
    if (pipelineStatus?.status === 'completed') {
      refetchDataCompare()
    }
  }, [pipelineStatus?.status])

  // Fetch lineage/history for the history tab
  const { data: lineageData, isLoading: isLineageLoading } = useQuery<LineageData>({
    queryKey: ['lineage', jobId],
    queryFn: async () => {
      const { data } = await axios.get(`/api/conversions/${jobId}/history`)
      return data
    },
    enabled: activeTab === 'history',
  })

  // Poll for Databricks job status when running
  useEffect(() => {
    if (!job) return

    // Only poll if job is in-progress
    const inProgressStatuses = ['parsing', 'converting', 'validating']
    if (!inProgressStatuses.includes(job.status)) {
      return
    }

    const pollStatus = async () => {
      try {
        const { data } = await axios.get(`/api/conversions/${jobId}/run-status`)
        setCurrentProgress(data.progress || 0)
        setCurrentStage(data.state || job.status)

        // Capture run page URL for linking to Databricks
        if (data.run_page_url) {
          setRunPageUrl(data.run_page_url)
        }

        // Add status message to events
        if (data.state_message) {
          setStatusEvents((prev) => {
            const newEvent = { message: `${data.state}: ${data.state_message}` }
            // Avoid duplicates
            if (prev.length > 0 && prev[prev.length - 1].message === newEvent.message) {
              return prev
            }
            return [...prev.slice(-20), newEvent]
          })
        }

        // If job completed or failed, refresh job data
        if (data.state === 'TERMINATED') {
          queryClient.invalidateQueries({ queryKey: ['conversion', jobId] })
        }
      } catch (error) {
        console.error('Failed to poll job status:', error)
      }
    }

    // Poll immediately and then every 3 seconds
    pollStatus()
    const interval = setInterval(pollStatus, 3000)

    return () => clearInterval(interval)
  }, [job?.status, jobId, queryClient])

  const runConversion = useMutation({
    mutationFn: async () => {
      const { data } = await axios.post(`/api/conversions/${jobId}/run`)
      return data
    },
    onSuccess: () => {
      setStatusEvents([])
      setCurrentProgress(0)
      queryClient.invalidateQueries({ queryKey: ['conversion', jobId] })
    },
  })

  const runValidation = useMutation({
    mutationFn: async () => {
      const { data } = await axios.post(`/api/validation/${jobId}/run`)
      return data
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['validation', jobId] })
      queryClient.invalidateQueries({ queryKey: ['conversion', jobId] })
    },
  })

  const summarizeIssues = useMutation({
    mutationFn: async () => {
      const { data } = await axios.post(`/api/conversions/${jobId}/summarize`)
      return data as IssueSummary
    },
    onSuccess: (data) => {
      setIssueSummary(data)
    },
  })

  const getRefinementPrompt = useMutation({
    mutationFn: async () => {
      const { data } = await axios.get(`/api/conversions/${jobId}/refinement-prompt`)
      return data
    },
    onSuccess: (data) => {
      setRefinementPrompt(data.refinement_prompt)
      setShowRefinementModal(true)
    },
  })

  const triggerRefinement = useMutation({
    mutationFn: async (params: { additional_instructions?: string; use_opus?: boolean }) => {
      const { data } = await axios.post(`/api/conversions/${jobId}/refine`, params)
      return data as RefineResult
    },
    onSuccess: (data) => {
      setRefineResult(data)
      setShowRefinementModal(false)
      queryClient.invalidateQueries({ queryKey: ['conversions'] })
    },
  })

  const runPipeline = useMutation({
    mutationFn: async () => {
      const { data } = await axios.post(`/api/validation/${jobId}/run-pipeline`)
      return data
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['pipelineStatus', jobId] })
    },
  })

  if (isLoading || !job) {
    return <div className="p-8">Loading...</div>
  }

  return (
    <div className="p-8">
      {/* Header */}
      <div className="mb-8">
        <Link to="/conversions" className="text-gray-600 hover:text-gray-900 flex items-center gap-1 mb-4">
          <ArrowLeft size={16} /> Back to Conversions
        </Link>

        {/* Refinement Banner - Show if this is a refinement of another job */}
        {job.metadata?.refinement === 'true' && job.metadata?.original_job_id && (
          <div className="mb-4 p-3 bg-purple-50 border border-purple-200 rounded-lg flex items-center gap-3">
            <GitBranch className="text-purple-600" size={20} />
            <div className="flex-1">
              <span className="text-purple-900 font-medium">Refinement Job</span>
              <span className="text-purple-700 mx-2">•</span>
              <Link
                to={`/conversions/${job.metadata.original_job_id}`}
                className="text-purple-600 hover:text-purple-800 hover:underline"
              >
                View Original Job
              </Link>
              {job.metadata.original_score && (
                <span className="text-purple-600 ml-2">(Original score: {job.metadata.original_score})</span>
              )}
            </div>
          </div>
        )}

        <div className="flex justify-between items-start">
          <div>
            <h1 className="text-2xl font-bold text-gray-900">{job.job_name}</h1>
            <p className="text-gray-600">{job.source_path}</p>
            {job.ai_model && (
              <p className="text-sm text-gray-500 mt-1">Model: {job.ai_model}</p>
            )}
          </div>
          <div className="flex gap-3">
            {job.status === 'pending' && (
              <button
                onClick={() => runConversion.mutate()}
                disabled={runConversion.isPending}
                className="px-4 py-2 bg-[#ff6600] text-white rounded-lg hover:bg-[#e55c00] flex items-center gap-2 disabled:opacity-50"
              >
                <Play size={16} /> Run Conversion
              </button>
            )}
            {job.status === 'completed' && (
              <>
                <button
                  onClick={() => runValidation.mutate()}
                  disabled={runValidation.isPending}
                  className="px-4 py-2 border rounded-lg hover:bg-gray-50 flex items-center gap-2"
                >
                  <RefreshCw size={16} /> Re-run Validation
                </button>
                <Link
                  to={`/promote?jobId=${jobId}`}
                  className="px-4 py-2 bg-green-600 text-white rounded-lg hover:bg-green-700 flex items-center gap-2"
                >
                  <Rocket size={16} /> Promote
                </Link>
              </>
            )}
          </div>
        </div>
      </div>

      {/* Progress Bar (when running) */}
      {['parsing', 'converting', 'validating'].includes(job.status) && (
        <div className="bg-white rounded-xl shadow-sm border p-6 mb-6">
          <div className="flex items-center justify-between mb-4">
            <div className="flex items-center gap-3">
              <Loader2 className="animate-spin text-[#ff6600]" size={24} />
              <div>
                <div className="font-medium">Conversion in Progress</div>
                <div className="text-sm text-gray-600">{currentStage || job.status}</div>
              </div>
            </div>
            {runPageUrl && (
              <a
                href={runPageUrl}
                target="_blank"
                rel="noopener noreferrer"
                className="text-sm text-blue-600 hover:underline"
              >
                View in Databricks
              </a>
            )}
          </div>
          <div className="relative h-4 bg-gray-200 rounded-full overflow-hidden">
            <div
              className="absolute left-0 top-0 h-full bg-[#ff6600] transition-all duration-500"
              style={{ width: `${currentProgress}%` }}
            />
          </div>
          <div className="mt-2 text-sm text-gray-600 text-right">{currentProgress}%</div>

          {/* Status Log */}
          {statusEvents.length > 0 && (
            <div className="mt-4 max-h-40 overflow-y-auto bg-gray-50 rounded-lg p-3 font-mono text-xs">
              {statusEvents.map((event, i) => (
                <div key={i} className={`py-1 ${event.error ? 'text-red-600' : 'text-gray-700'}`}>
                  {event.message || event.status || event.error}
                </div>
              ))}
            </div>
          )}
        </div>
      )}

      {/* Tab Navigation */}
      <div className="mb-6 border-b border-gray-200">
        <nav className="flex gap-4" aria-label="Tabs">
          <button
            onClick={() => setActiveTab('overview')}
            className={`px-4 py-3 text-sm font-medium border-b-2 transition-colors flex items-center gap-2 ${
              activeTab === 'overview'
                ? 'border-[#ff6600] text-[#ff6600]'
                : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
            }`}
          >
            <LayoutDashboard size={16} />
            Overview
          </button>
          <button
            onClick={() => setActiveTab('code')}
            disabled={job.status !== 'completed'}
            className={`px-4 py-3 text-sm font-medium border-b-2 transition-colors flex items-center gap-2 ${
              activeTab === 'code'
                ? 'border-[#ff6600] text-[#ff6600]'
                : job.status !== 'completed'
                ? 'border-transparent text-gray-300 cursor-not-allowed'
                : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
            }`}
          >
            <Code2 size={16} />
            Code
          </button>
          <button
            onClick={() => setActiveTab('history')}
            className={`px-4 py-3 text-sm font-medium border-b-2 transition-colors flex items-center gap-2 ${
              activeTab === 'history'
                ? 'border-[#ff6600] text-[#ff6600]'
                : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
            }`}
          >
            <History size={16} />
            History
          </button>
          <button
            onClick={() => setActiveTab('comparisons')}
            disabled={job.status !== 'completed'}
            className={`px-4 py-3 text-sm font-medium border-b-2 transition-colors flex items-center gap-2 ${
              activeTab === 'comparisons'
                ? 'border-[#ff6600] text-[#ff6600]'
                : job.status !== 'completed'
                ? 'border-transparent text-gray-300 cursor-not-allowed'
                : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
            }`}
          >
            <GitCompare size={16} />
            Comparisons
          </button>
        </nav>
      </div>

      {/* Tab Content */}
      {activeTab === 'overview' && (
        <>
      {/* Status and Score Cards */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-8">
        <div className="bg-white rounded-xl shadow-sm border p-6">
          <h3 className="text-sm font-medium text-gray-500 mb-2">Status</h3>
          <StatusBadge status={job.status} size="lg" />
          {job.error_message && (
            <p className="mt-2 text-sm text-red-600">{job.error_message}</p>
          )}
        </div>

        <div className="bg-white rounded-xl shadow-sm border p-6">
          <h3 className="text-sm font-medium text-gray-500 mb-2">Quality Score</h3>
          {job.quality_score !== null ? (
            <ScoreCard score={job.quality_score} />
          ) : (
            <span className="text-gray-400">Not validated yet</span>
          )}
        </div>

        <div className="bg-white rounded-xl shadow-sm border p-6">
          <h3 className="text-sm font-medium text-gray-500 mb-2">Timeline</h3>
          <div className="space-y-2 text-sm">
            <div className="flex justify-between">
              <span className="text-gray-600">Created:</span>
              <span>{new Date(job.created_at).toLocaleString()}</span>
            </div>
            {job.completed_at && (
              <div className="flex justify-between">
                <span className="text-gray-600">Completed:</span>
                <span>{new Date(job.completed_at).toLocaleString()}</span>
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Conversion Settings */}
      {(job.ai_model || job.conversion_instructions || (job.reference_files && job.reference_files.length > 0)) && (
        <div className="bg-white rounded-xl shadow-sm border p-6 mb-8">
          <h2 className="text-lg font-semibold mb-4">Conversion Settings</h2>
          <div className="space-y-4">
            {job.ai_model && (
              <div>
                <span className="text-sm font-medium text-gray-500">AI Model:</span>
                <span className="ml-2 text-sm">{job.ai_model}</span>
              </div>
            )}
            {job.conversion_instructions && (
              <div>
                <span className="text-sm font-medium text-gray-500 block mb-1">Instructions:</span>
                <div className="bg-gray-50 rounded-lg p-3 text-sm whitespace-pre-wrap">
                  {job.conversion_instructions}
                </div>
              </div>
            )}
            {job.reference_files && job.reference_files.length > 0 && (
              <div>
                <span className="text-sm font-medium text-gray-500 block mb-2">Reference Files:</span>
                <div className="flex flex-wrap gap-2">
                  {job.reference_files.map((ref) => (
                    <span key={ref.file_id} className="inline-flex items-center gap-1 px-3 py-1 bg-gray-100 rounded-full text-sm">
                      <FileText size={14} />
                      {ref.file_name}
                      <span className="text-xs text-gray-500">({ref.file_type})</span>
                    </span>
                  ))}
                </div>
              </div>
            )}
          </div>
        </div>
      )}

      {/* Validation Pipeline Banner (auto-started) */}
      {job.status === 'completed' && pipelineStatus && pipelineStatus.status !== 'not_started' && (
        <div className={`mb-6 p-4 rounded-xl border flex items-center justify-between ${
          pipelineStatus.status === 'running' ? 'bg-blue-50 border-blue-200' :
          pipelineStatus.status === 'completed' ? 'bg-green-50 border-green-200' :
          pipelineStatus.status === 'failed' ? 'bg-red-50 border-red-200' :
          'bg-gray-50 border-gray-200'
        }`}>
          <div className="flex items-center gap-3">
            {pipelineStatus.status === 'running' ? (
              <Loader2 size={20} className="animate-spin text-blue-600" />
            ) : pipelineStatus.status === 'completed' ? (
              <CheckCircle2 size={20} className="text-green-600" />
            ) : pipelineStatus.status === 'failed' ? (
              <XCircle size={20} className="text-red-600" />
            ) : (
              <FlaskConical size={20} className="text-gray-600" />
            )}
            <div>
              <span className="font-medium text-sm">
                {pipelineStatus.status === 'running' ? 'Validation pipeline running...' :
                 pipelineStatus.status === 'completed' ? 'Validation pipeline completed' :
                 pipelineStatus.status === 'failed' ? 'Validation pipeline failed' :
                 'Validation pipeline'}
              </span>
              {pipelineStatus.error && (
                <p className="text-xs text-red-600 mt-0.5">{pipelineStatus.error}</p>
              )}
            </div>
          </div>
          <button
            onClick={() => setActiveTab('comparisons')}
            className="text-sm text-blue-600 hover:underline"
          >
            View Details
          </button>
        </div>
      )}

      {/* Quality Report */}
      {job.status === 'completed' && (
        <div className="mb-8">
          <QualityReport report={job.quality_report || null} />

          {/* AI-Powered Actions - Always show for completed jobs */}
          <div className="mt-4 flex flex-wrap gap-3">
            <button
              onClick={() => summarizeIssues.mutate()}
              disabled={summarizeIssues.isPending}
              className="px-4 py-2 bg-purple-600 text-white rounded-lg hover:bg-purple-700 flex items-center gap-2 disabled:opacity-50"
            >
              {summarizeIssues.isPending ? (
                <Loader2 size={16} className="animate-spin" />
              ) : (
                <Sparkles size={16} />
              )}
              Summarize Issues (Haiku)
            </button>

            <button
              onClick={() => getRefinementPrompt.mutate()}
              disabled={getRefinementPrompt.isPending}
              className="px-4 py-2 bg-orange-600 text-white rounded-lg hover:bg-orange-700 flex items-center gap-2 disabled:opacity-50"
            >
              {getRefinementPrompt.isPending ? (
                <Loader2 size={16} className="animate-spin" />
              ) : (
                <Wrench size={16} />
              )}
              Refine Conversion
            </button>
          </div>

          {/* Issue Summary Card */}
          {issueSummary && (
            <div className={`mt-4 rounded-xl border p-6 ${
              issueSummary.severity === 'error' ? 'bg-red-50 border-red-200' :
              issueSummary.severity === 'warning' ? 'bg-yellow-50 border-yellow-200' :
              'bg-blue-50 border-blue-200'
            }`}>
              <div className="flex items-start gap-3">
                <AlertCircle className={`mt-0.5 ${
                  issueSummary.severity === 'error' ? 'text-red-600' :
                  issueSummary.severity === 'warning' ? 'text-yellow-600' :
                  'text-blue-600'
                }`} size={24} />
                <div className="flex-1">
                  <h3 className="font-semibold text-gray-900 mb-2">AI Summary</h3>
                  <p className="text-gray-700 whitespace-pre-wrap">{issueSummary.summary}</p>
                  {issueSummary.issues.length > 0 && (
                    <div className="mt-3 text-sm text-gray-600">
                      {issueSummary.issue_count} issue{issueSummary.issue_count !== 1 ? 's' : ''} found
                    </div>
                  )}
                </div>
              </div>
            </div>
          )}

          {/* Refinement Success Message */}
          {refineResult && (
            <div className="mt-4 rounded-xl border border-green-200 bg-green-50 p-6">
              <div className="flex items-start gap-3">
                <CheckCircle2 className="text-green-600 mt-0.5" size={24} />
                <div className="flex-1">
                  <h3 className="font-semibold text-gray-900 mb-2">Refinement Job Started</h3>
                  <p className="text-gray-700">{refineResult.message}</p>
                  <div className="mt-3 flex gap-3">
                    <Link
                      to={`/conversions/${refineResult.refinement_job_id}`}
                      className="text-blue-600 hover:underline text-sm"
                    >
                      View Refinement Job
                    </Link>
                    <span className="text-gray-400 text-sm">Model: {refineResult.ai_model}</span>
                  </div>
                </div>
              </div>
            </div>
          )}
        </div>
      )}

      {/* Refinement Modal */}
      {showRefinementModal && (
        <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50 p-4">
          <div className="bg-white rounded-xl shadow-xl max-w-2xl w-full max-h-[90vh] overflow-y-auto">
            <div className="p-6 border-b">
              <h2 className="text-xl font-semibold flex items-center gap-2">
                <Wrench size={24} />
                Refine Conversion
              </h2>
              <p className="text-gray-600 mt-1">
                Launch a new conversion job with instructions to fix issues
              </p>
            </div>

            <div className="p-6 space-y-4">
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Refinement Instructions
                </label>
                <textarea
                  value={refinementPrompt}
                  onChange={(e) => setRefinementPrompt(e.target.value)}
                  rows={10}
                  className="w-full border rounded-lg p-3 font-mono text-sm"
                  placeholder="Instructions for the AI to fix issues..."
                />
              </div>

              <div className="flex items-center gap-2">
                <input
                  type="checkbox"
                  id="use_opus"
                  className="rounded"
                />
                <label htmlFor="use_opus" className="text-sm text-gray-700">
                  Use Claude Opus (more powerful, slower)
                </label>
              </div>
            </div>

            <div className="p-6 border-t flex justify-end gap-3">
              <button
                onClick={() => setShowRefinementModal(false)}
                className="px-4 py-2 border rounded-lg hover:bg-gray-50"
              >
                Cancel
              </button>
              <button
                onClick={() => {
                  const useOpus = (document.getElementById('use_opus') as HTMLInputElement)?.checked
                  triggerRefinement.mutate({
                    additional_instructions: refinementPrompt,
                    use_opus: useOpus,
                  })
                }}
                disabled={triggerRefinement.isPending}
                className="px-4 py-2 bg-orange-600 text-white rounded-lg hover:bg-orange-700 flex items-center gap-2 disabled:opacity-50"
              >
                {triggerRefinement.isPending ? (
                  <Loader2 size={16} className="animate-spin" />
                ) : (
                  <Play size={16} />
                )}
                Start Refinement Job
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Output Files */}
      {job.output_path && (
        <div className="bg-white rounded-xl shadow-sm border p-6 mb-8">
          <h2 className="text-lg font-semibold mb-4">Output Location</h2>
          <div className="bg-gray-50 rounded-lg p-4">
            <code className="text-sm">{job.output_path}</code>
          </div>
        </div>
      )}

      {/* Validation Results (legacy format, kept for backward compatibility) */}
      {validations && validations.length > 0 && (
        <div className="bg-white rounded-xl shadow-sm border p-6">
          <h2 className="text-lg font-semibold mb-4">Validation Results</h2>
          <div className="space-y-2">
            {validations.map((v) => (
              <div
                key={v.validation_id}
                className={`flex items-start gap-3 p-3 rounded-lg ${
                  v.passed ? 'bg-green-50' : 'bg-red-50'
                }`}
              >
                {v.passed ? (
                  <CheckCircle2 className="text-green-600 mt-0.5" size={20} />
                ) : (
                  <XCircle className="text-red-600 mt-0.5" size={20} />
                )}
                <div className="flex-1">
                  <div className="font-medium text-gray-900">{v.check_name}</div>
                  <div className="text-sm text-gray-600">{v.message}</div>
                  {v.expected && (
                    <div className="text-xs text-gray-500 mt-1">
                      Expected: {v.expected} | Actual: {v.actual}
                    </div>
                  )}
                </div>
                <span className="px-2 py-1 text-xs font-medium bg-white rounded uppercase">
                  {v.category}
                </span>
              </div>
            ))}
          </div>
        </div>
      )}
        </>
      )}

      {/* Code Tab */}
      {activeTab === 'code' && (
        <>
          {job.status === 'completed' && codeCompare ? (
            <CodeCompare
              sourceFile={codeCompare.source_file}
              sourceContent={codeCompare.source_content}
              outputFiles={codeCompare.output_files}
            />
          ) : (
            <div className="bg-white rounded-xl shadow-sm border p-8 text-center">
              <Code2 className="mx-auto text-gray-400 mb-4" size={48} />
              <h3 className="text-lg font-medium text-gray-900 mb-2">Code Not Available</h3>
              <p className="text-gray-500">
                Code comparison will be available once the conversion is complete.
              </p>
            </div>
          )}
        </>
      )}

      {/* History Tab */}
      {activeTab === 'history' && (
        <LineageTree
          original={lineageData?.original || null}
          refinements={lineageData?.refinements || []}
          currentJobId={jobId!}
          isLoading={isLineageLoading}
        />
      )}

      {/* Comparisons Tab */}
      {activeTab === 'comparisons' && (
        <>
          {job.status === 'completed' ? (
            <>
              {/* Validation Pipeline Trigger & Status */}
              <div className="bg-white rounded-xl shadow-sm border p-6 mb-6">
                <div className="flex items-center justify-between">
                  <div>
                    <h2 className="text-lg font-semibold flex items-center gap-2">
                      <FlaskConical size={20} />
                      Validation Pipeline
                    </h2>
                    <p className="text-sm text-gray-500 mt-1">
                      Run the full 3-step validation: Simulate Original &rarr; Run Converted &rarr; Compare Data
                    </p>
                  </div>
                  <button
                    onClick={() => runPipeline.mutate()}
                    disabled={runPipeline.isPending || pipelineStatus?.status === 'running'}
                    className="px-4 py-2 bg-[#ff6600] text-white rounded-lg hover:bg-[#e55c00] flex items-center gap-2 disabled:opacity-50"
                  >
                    {(runPipeline.isPending || pipelineStatus?.status === 'running') ? (
                      <Loader2 size={16} className="animate-spin" />
                    ) : (
                      <Play size={16} />
                    )}
                    {pipelineStatus?.status === 'running' ? 'Pipeline Running...' : 'Run Full Validation Pipeline'}
                  </button>
                </div>

                {/* Pipeline Steps Progress */}
                {pipelineStatus && pipelineStatus.status !== 'not_started' && (
                  <div className="mt-4">
                    <div className="flex items-center gap-2 mb-3">
                      {(['original_simulator', 'converted_runner', 'data_comparator'] as const).map((step, i) => {
                        const stepData = pipelineStatus.steps?.[step]
                        const stepLabels = ['Simulate Original', 'Run Converted', 'Compare Data']
                        const status = stepData?.status || 'pending'

                        return (
                          <div key={step} className="flex items-center gap-2">
                            {i > 0 && <div className="w-8 h-px bg-gray-300" />}
                            <div className={`flex items-center gap-1.5 px-3 py-1.5 rounded-full text-xs font-medium ${
                              status === 'SUCCESS' ? 'bg-green-100 text-green-800' :
                              status === 'running' ? 'bg-blue-100 text-blue-800' :
                              status === 'FAILED' ? 'bg-red-100 text-red-800' :
                              'bg-gray-100 text-gray-600'
                            }`}>
                              {status === 'SUCCESS' ? <CheckCircle2 size={12} /> :
                               status === 'running' ? <Loader2 size={12} className="animate-spin" /> :
                               status === 'FAILED' ? <XCircle size={12} /> :
                               <span className="w-3 h-3 rounded-full bg-gray-300 inline-block" />}
                              {stepLabels[i]}
                            </div>
                          </div>
                        )
                      })}
                    </div>

                    {pipelineStatus.status === 'completed' && (
                      <div className="p-3 bg-green-50 border border-green-200 rounded-lg text-sm text-green-800 flex items-center gap-2">
                        <CheckCircle2 size={16} />
                        Pipeline completed successfully! Data comparison results are shown below.
                      </div>
                    )}
                    {pipelineStatus.status === 'failed' && (
                      <div className="p-3 bg-red-50 border border-red-200 rounded-lg text-sm text-red-800 flex items-center gap-2">
                        <XCircle size={16} />
                        Pipeline failed: {pipelineStatus.error || 'Unknown error'}
                      </div>
                    )}
                    {pipelineStatus.status === 'running' && (
                      <div className="p-3 bg-blue-50 border border-blue-200 rounded-lg text-sm text-blue-800 flex items-center gap-2">
                        <Loader2 size={16} className="animate-spin" />
                        {pipelineStatus.message || 'Pipeline is running...'}
                      </div>
                    )}

                    {/* Links to Databricks job runs */}
                    {pipelineStatus.steps && Object.entries(pipelineStatus.steps).some(([, s]) => s.run_page_url) && (
                      <div className="mt-2 flex gap-3 text-xs">
                        {Object.entries(pipelineStatus.steps).map(([key, s]) =>
                          s.run_page_url ? (
                            <a key={key} href={s.run_page_url} target="_blank" rel="noopener noreferrer" className="text-blue-600 hover:underline">
                              View {key.replace(/_/g, ' ')} run
                            </a>
                          ) : null
                        )}
                      </div>
                    )}
                  </div>
                )}
              </div>

              {/* Data Compare */}
              <DataCompare
                original={dataCompare?.original || null}
                converted={dataCompare?.converted || null}
                comparison={dataCompare?.comparison}
                onRefresh={() => refetchDataCompare()}
                isLoading={isDataCompareLoading}
              />
            </>
          ) : (
            <div className="bg-white rounded-xl shadow-sm border p-8 text-center">
              <GitCompare className="mx-auto text-gray-400 mb-4" size={48} />
              <h3 className="text-lg font-medium text-gray-900 mb-2">Comparisons Not Available</h3>
              <p className="text-gray-500">
                Data comparisons will be available once the conversion is complete.
              </p>
            </div>
          )}
        </>
      )}
    </div>
  )
}

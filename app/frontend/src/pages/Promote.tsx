import { useState } from 'react'
import { useSearchParams } from 'react-router-dom'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import {
  Rocket,
  CheckCircle2,
  XCircle,
  Clock,
  ArrowRight,
} from 'lucide-react'
import axios from 'axios'

interface PromotionHistory {
  promotion_id: string
  job_id: string
  from_environment: string
  to_environment: string
  promoted_by: string
  promoted_at: string
  approval_status: string
  approver: string | null
  approved_at: string | null
  rejection_reason: string | null
}

interface PendingApproval extends PromotionHistory {
  job_name: string
  quality_score: number
}

export default function Promote() {
  const [searchParams] = useSearchParams()
  const preselectedJobId = searchParams.get('jobId')
  const queryClient = useQueryClient()

  const [selectedJobId, setSelectedJobId] = useState(preselectedJobId || '')
  const [targetEnv, setTargetEnv] = useState<'qa' | 'prod'>('qa')
  const [notes, setNotes] = useState('')

  // Get completed jobs for selection
  const { data: jobs } = useQuery({
    queryKey: ['conversions', 'completed'],
    queryFn: async () => {
      const { data } = await axios.get('/api/conversions', { params: { status: 'completed' } })
      return data
    },
  })

  // Get pending approvals
  const { data: pendingApprovals } = useQuery<PendingApproval[]>({
    queryKey: ['pending-approvals'],
    queryFn: async () => {
      // In production, this would call a dedicated endpoint
      // For now, aggregate from all jobs
      const allHistory: PromotionHistory[] = []
      for (const job of jobs || []) {
        try {
          const { data } = await axios.get(`/api/workflows/${job.job_id}/history`)
          allHistory.push(...data.filter((h: PromotionHistory) => h.approval_status === 'pending'))
        } catch {}
      }
      return allHistory as PendingApproval[]
    },
    enabled: !!jobs,
  })

  const promoteMutation = useMutation({
    mutationFn: async () => {
      const { data } = await axios.post('/api/workflows/promote', {
        job_id: selectedJobId,
        to_environment: targetEnv,
        notes,
      })
      return data
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['pending-approvals'] })
      setSelectedJobId('')
      setNotes('')
    },
  })

  const approveMutation = useMutation({
    mutationFn: async ({ promotionId, approved, reason }: { promotionId: string; approved: boolean; reason?: string }) => {
      const { data } = await axios.post(`/api/workflows/promote/${promotionId}/approve`, null, {
        params: { approved, rejection_reason: reason },
      })
      return data
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['pending-approvals'] })
    },
  })

  return (
    <div className="p-8">
      <div className="mb-8">
        <h1 className="text-2xl font-bold text-gray-900">Promote to Production</h1>
        <p className="text-gray-600">Move conversions through dev → QA → prod workflow</p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
        {/* Request Promotion */}
        <div className="bg-white rounded-xl shadow-sm border p-6">
          <h2 className="text-lg font-semibold mb-4">Request Promotion</h2>

          <div className="space-y-4">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Select Conversion
              </label>
              <select
                value={selectedJobId}
                onChange={(e) => setSelectedJobId(e.target.value)}
                className="w-full px-4 py-2 border rounded-lg"
              >
                <option value="">Choose a completed conversion...</option>
                {jobs?.map((job: any) => (
                  <option key={job.job_id} value={job.job_id}>
                    {job.job_name} ({(job.quality_score * 100).toFixed(0)}%)
                  </option>
                ))}
              </select>
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Target Environment
              </label>
              <div className="flex gap-4">
                <label className="flex items-center gap-2 cursor-pointer">
                  <input
                    type="radio"
                    checked={targetEnv === 'qa'}
                    onChange={() => setTargetEnv('qa')}
                    className="text-[#ff6600]"
                  />
                  <span>QA</span>
                </label>
                <label className="flex items-center gap-2 cursor-pointer">
                  <input
                    type="radio"
                    checked={targetEnv === 'prod'}
                    onChange={() => setTargetEnv('prod')}
                    className="text-[#ff6600]"
                  />
                  <span>Production</span>
                </label>
              </div>
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Notes (optional)
              </label>
              <textarea
                value={notes}
                onChange={(e) => setNotes(e.target.value)}
                rows={3}
                placeholder="Add any notes for the approver..."
                className="w-full px-4 py-2 border rounded-lg"
              />
            </div>

            <button
              onClick={() => promoteMutation.mutate()}
              disabled={!selectedJobId || promoteMutation.isPending}
              className="w-full px-4 py-3 bg-[#ff6600] text-white rounded-lg hover:bg-[#e55c00] disabled:opacity-50 flex items-center justify-center gap-2"
            >
              <Rocket size={20} />
              {promoteMutation.isPending ? 'Requesting...' : `Request Promotion to ${targetEnv.toUpperCase()}`}
            </button>
          </div>

          {promoteMutation.isSuccess && (
            <div className="mt-4 p-4 bg-green-50 text-green-700 rounded-lg flex items-center gap-2">
              <CheckCircle2 size={20} />
              Promotion request submitted! Awaiting approval.
            </div>
          )}
        </div>

        {/* Pending Approvals */}
        <div className="bg-white rounded-xl shadow-sm border p-6">
          <h2 className="text-lg font-semibold mb-4">Pending Approvals</h2>

          {!pendingApprovals || pendingApprovals.length === 0 ? (
            <div className="text-center py-8 text-gray-500">
              <Clock size={32} className="mx-auto mb-2 text-gray-400" />
              No pending approvals
            </div>
          ) : (
            <div className="space-y-4">
              {pendingApprovals.map((approval) => (
                <div key={approval.promotion_id} className="border rounded-lg p-4">
                  <div className="flex items-center justify-between mb-2">
                    <span className="font-medium">{approval.job_name || approval.job_id}</span>
                    <div className="flex items-center gap-2 text-sm text-gray-600">
                      <span className="uppercase">{approval.from_environment}</span>
                      <ArrowRight size={16} />
                      <span className="uppercase font-medium text-[#ff6600]">{approval.to_environment}</span>
                    </div>
                  </div>
                  <div className="text-sm text-gray-500 mb-3">
                    Requested by {approval.promoted_by} on {new Date(approval.promoted_at).toLocaleString()}
                  </div>
                  <div className="flex gap-2">
                    <button
                      onClick={() => approveMutation.mutate({ promotionId: approval.promotion_id, approved: true })}
                      className="flex-1 px-3 py-2 bg-green-600 text-white rounded-lg hover:bg-green-700 flex items-center justify-center gap-1"
                    >
                      <CheckCircle2 size={16} /> Approve
                    </button>
                    <button
                      onClick={() => {
                        const reason = prompt('Rejection reason:')
                        if (reason) {
                          approveMutation.mutate({ promotionId: approval.promotion_id, approved: false, reason })
                        }
                      }}
                      className="flex-1 px-3 py-2 bg-red-600 text-white rounded-lg hover:bg-red-700 flex items-center justify-center gap-1"
                    >
                      <XCircle size={16} /> Reject
                    </button>
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      </div>

      {/* Promotion Workflow Diagram */}
      <div className="mt-8 bg-white rounded-xl shadow-sm border p-6">
        <h2 className="text-lg font-semibold mb-4">Promotion Workflow</h2>
        <div className="flex items-center justify-center gap-4 py-4">
          <div className="text-center">
            <div className="w-24 h-24 rounded-full bg-blue-100 flex items-center justify-center mx-auto mb-2">
              <span className="text-2xl font-bold text-blue-600">DEV</span>
            </div>
            <div className="text-sm text-gray-600">Auto-deploy on<br/>successful conversion</div>
          </div>
          <ArrowRight size={32} className="text-gray-400" />
          <div className="text-center">
            <div className="w-24 h-24 rounded-full bg-yellow-100 flex items-center justify-center mx-auto mb-2">
              <span className="text-2xl font-bold text-yellow-600">QA</span>
            </div>
            <div className="text-sm text-gray-600">Requires<br/>QA approval</div>
          </div>
          <ArrowRight size={32} className="text-gray-400" />
          <div className="text-center">
            <div className="w-24 h-24 rounded-full bg-green-100 flex items-center justify-center mx-auto mb-2">
              <span className="text-2xl font-bold text-green-600">PROD</span>
            </div>
            <div className="text-sm text-gray-600">Requires<br/>designated approver</div>
          </div>
        </div>
      </div>
    </div>
  )
}

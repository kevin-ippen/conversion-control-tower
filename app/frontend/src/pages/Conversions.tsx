import { useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import { Link } from 'react-router-dom'
import {
  FileCode2,
  ChevronRight,
  RefreshCw,
} from 'lucide-react'
import axios from 'axios'
import StatusBadge from '../components/StatusBadge'
import ScoreCard from '../components/ScoreCard'

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
}

export default function Conversions() {
  const [statusFilter, setStatusFilter] = useState<string>('')

  const { data: jobs, isLoading, refetch } = useQuery<ConversionJob[]>({
    queryKey: ['conversions', statusFilter],
    queryFn: async () => {
      const params = statusFilter ? { status: statusFilter } : {}
      const { data } = await axios.get('/api/conversions', { params })
      return data
    },
  })

  return (
    <div className="p-8">
      <div className="flex justify-between items-center mb-8">
        <div>
          <h1 className="text-2xl font-bold text-gray-900">Conversions</h1>
          <p className="text-gray-600">AI-powered SQL to Databricks migrations</p>
        </div>
        <div className="flex gap-3">
          <select
            value={statusFilter}
            onChange={(e) => setStatusFilter(e.target.value)}
            className="px-4 py-2 border rounded-lg bg-white"
          >
            <option value="">All Status</option>
            <option value="pending">Pending</option>
            <option value="parsing">Parsing</option>
            <option value="converting">Converting</option>
            <option value="validating">Validating</option>
            <option value="completed">Completed</option>
            <option value="failed">Failed</option>
          </select>
          <button
            onClick={() => refetch()}
            className="p-2 border rounded-lg hover:bg-gray-50"
          >
            <RefreshCw size={20} />
          </button>
          <Link
            to="/new"
            className="px-4 py-2 bg-[#ff6600] text-white rounded-lg hover:bg-[#e55c00] flex items-center gap-2"
          >
            New Conversion
          </Link>
        </div>
      </div>

      {isLoading ? (
        <div className="text-center py-12">Loading...</div>
      ) : jobs?.length === 0 ? (
        <div className="text-center py-12 bg-white rounded-xl border">
          <FileCode2 size={48} className="mx-auto text-gray-400 mb-4" />
          <h3 className="text-lg font-medium text-gray-900">No conversions yet</h3>
          <p className="text-gray-600 mb-4">Upload a file to start your first conversion</p>
          <Link
            to="/new"
            className="inline-flex px-4 py-2 bg-[#ff6600] text-white rounded-lg hover:bg-[#e55c00]"
          >
            Upload File
          </Link>
        </div>
      ) : (
        <div className="bg-white rounded-xl shadow-sm border overflow-hidden">
          <table className="w-full">
            <thead className="bg-gray-50 border-b">
              <tr>
                <th className="text-left px-6 py-3 text-sm font-medium text-gray-500">Job</th>
                <th className="text-left px-6 py-3 text-sm font-medium text-gray-500">Type</th>
                <th className="text-left px-6 py-3 text-sm font-medium text-gray-500">Status</th>
                <th className="text-left px-6 py-3 text-sm font-medium text-gray-500">Score</th>
                <th className="text-left px-6 py-3 text-sm font-medium text-gray-500">Created</th>
                <th className="px-6 py-3"></th>
              </tr>
            </thead>
            <tbody className="divide-y">
              {jobs?.map((job) => (
                <tr key={job.job_id} className="hover:bg-gray-50">
                  <td className="px-6 py-4">
                    <div>
                      <div className="font-medium text-gray-900">{job.job_name}</div>
                      <div className="text-sm text-gray-500 truncate max-w-xs">{job.source_path}</div>
                    </div>
                  </td>
                  <td className="px-6 py-4">
                    <span className="px-2 py-1 text-xs font-medium bg-gray-100 rounded uppercase">
                      {job.source_type.replace('_', ' ')}
                    </span>
                  </td>
                  <td className="px-6 py-4">
                    <StatusBadge status={job.status} />
                  </td>
                  <td className="px-6 py-4">
                    {job.quality_score !== null ? (
                      <ScoreCard score={job.quality_score} compact />
                    ) : (
                      <span className="text-gray-400">—</span>
                    )}
                  </td>
                  <td className="px-6 py-4 text-sm text-gray-500">
                    {new Date(job.created_at).toLocaleDateString()}
                  </td>
                  <td className="px-6 py-4">
                    <Link
                      to={`/conversions/${job.job_id}`}
                      className="text-[#ff6600] hover:underline flex items-center gap-1"
                    >
                      View <ChevronRight size={16} />
                    </Link>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  )
}

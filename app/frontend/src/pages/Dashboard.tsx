import { useQuery } from '@tanstack/react-query'
import {
  CheckCircle2,
  Clock,
  FileCode2,
  TrendingUp,
} from 'lucide-react'
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, PieChart, Pie, Cell } from 'recharts'
import axios from 'axios'

interface OverviewMetrics {
  total_jobs: number
  completed_jobs: number
  failed_jobs: number
  in_progress_jobs: number
  average_quality_score: number
  jobs_by_status: Record<string, number>
  jobs_by_source_type: Record<string, number>
}

interface QualityDistribution {
  grade_a: number
  grade_b: number
  grade_c: number
  grade_d: number
  grade_f: number
}

export default function Dashboard() {
  const { data: metrics, isLoading: metricsLoading } = useQuery<OverviewMetrics>({
    queryKey: ['analytics', 'overview'],
    queryFn: async () => {
      const { data } = await axios.get('/api/analytics/overview')
      return data
    },
  })

  const { data: quality } = useQuery<QualityDistribution>({
    queryKey: ['analytics', 'quality'],
    queryFn: async () => {
      const { data } = await axios.get('/api/analytics/quality')
      return data
    },
  })

  const qualityData = quality ? [
    { name: 'A (90%+)', value: quality.grade_a, color: '#22c55e' },
    { name: 'B (80-89%)', value: quality.grade_b, color: '#84cc16' },
    { name: 'C (70-79%)', value: quality.grade_c, color: '#eab308' },
    { name: 'D (60-69%)', value: quality.grade_d, color: '#f97316' },
    { name: 'F (<60%)', value: quality.grade_f, color: '#ef4444' },
  ].filter(d => d.value > 0) : []

  if (metricsLoading) {
    return <div className="p-8">Loading...</div>
  }

  return (
    <div className="p-8">
      <div className="mb-8">
        <h1 className="text-2xl font-bold text-gray-900">Conversion Control Tower</h1>
        <p className="text-gray-600">AI-powered code conversion to Databricks</p>
      </div>

      {/* Stats Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
        <StatCard
          title="Total Conversions"
          value={metrics?.total_jobs ?? 0}
          icon={<FileCode2 className="text-blue-500" />}
          bgColor="bg-blue-50"
        />
        <StatCard
          title="Completed"
          value={metrics?.completed_jobs ?? 0}
          icon={<CheckCircle2 className="text-green-500" />}
          bgColor="bg-green-50"
        />
        <StatCard
          title="In Progress"
          value={metrics?.in_progress_jobs ?? 0}
          icon={<Clock className="text-yellow-500" />}
          bgColor="bg-yellow-50"
        />
        <StatCard
          title="Avg Quality Score"
          value={`${((metrics?.average_quality_score ?? 0) * 100).toFixed(0)}%`}
          icon={<TrendingUp className="text-purple-500" />}
          bgColor="bg-purple-50"
        />
      </div>

      {/* Charts */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Status Distribution */}
        <div className="bg-white rounded-xl shadow-sm border p-6">
          <h2 className="text-lg font-semibold mb-4">Conversions by Status</h2>
          <div className="h-64">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart
                data={Object.entries(metrics?.jobs_by_status ?? {}).map(([name, value]) => ({
                  name: name.charAt(0).toUpperCase() + name.slice(1),
                  count: value,
                }))}
              >
                <XAxis dataKey="name" />
                <YAxis />
                <Tooltip />
                <Bar dataKey="count" fill="#ff6600" radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Quality Distribution */}
        <div className="bg-white rounded-xl shadow-sm border p-6">
          <h2 className="text-lg font-semibold mb-4">Quality Score Distribution</h2>
          <div className="h-64">
            <ResponsiveContainer width="100%" height="100%">
              <PieChart>
                <Pie
                  data={qualityData}
                  cx="50%"
                  cy="50%"
                  innerRadius={60}
                  outerRadius={80}
                  dataKey="value"
                  label={({ name, value }) => `${name}: ${value}`}
                >
                  {qualityData.map((entry, index) => (
                    <Cell key={index} fill={entry.color} />
                  ))}
                </Pie>
                <Tooltip />
              </PieChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>

      {/* Source Type Breakdown */}
      <div className="mt-6 bg-white rounded-xl shadow-sm border p-6">
        <h2 className="text-lg font-semibold mb-4">Conversions by Source Type</h2>
        <div className="grid grid-cols-3 gap-4">
          {Object.entries(metrics?.jobs_by_source_type ?? {}).map(([type, count]) => (
            <div key={type} className="text-center p-4 bg-gray-50 rounded-lg">
              <div className="text-2xl font-bold text-gray-900">{count}</div>
              <div className="text-sm text-gray-600 uppercase">{type.replace('_', ' ')}</div>
            </div>
          ))}
        </div>
      </div>
    </div>
  )
}

function StatCard({
  title,
  value,
  icon,
  bgColor,
}: {
  title: string
  value: string | number
  icon: React.ReactNode
  bgColor: string
}) {
  return (
    <div className="bg-white rounded-xl shadow-sm border p-6">
      <div className="flex items-center gap-4">
        <div className={`p-3 rounded-lg ${bgColor}`}>{icon}</div>
        <div>
          <p className="text-sm text-gray-600">{title}</p>
          <p className="text-2xl font-bold text-gray-900">{value}</p>
        </div>
      </div>
    </div>
  )
}

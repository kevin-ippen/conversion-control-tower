import { useState } from 'react'
import { Table2, ArrowLeftRight, CheckCircle2, XCircle, AlertTriangle, RefreshCw, FlaskConical, Cpu } from 'lucide-react'

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
  schema?: string
  table?: string
  row_count: number
  column_count: number
  columns: ColumnSchema[]
  sample: DataSample
}

interface DataCompareProps {
  original: DataSource | null  // Expected output (from SSIS simulation)
  converted: DataSource | null // Actual output (from Databricks code)
  comparison?: {
    row_count_match: boolean
    schema_match: boolean
    sample_match_rate: number
    mismatched_columns: string[]
    summary: string
  }
  onRefresh?: () => void
  isLoading?: boolean
}

export default function DataCompare({
  original: expected,  // Rename for clarity - "expected" output from SSIS simulation
  converted: actual,   // Rename for clarity - "actual" output from Databricks
  comparison,
  onRefresh,
  isLoading
}: DataCompareProps) {
  const [activeTab, setActiveTab] = useState<'schema' | 'sample' | 'diff'>('sample')

  if (!expected && !actual) {
    return (
      <div className="bg-white rounded-xl shadow-sm border">
        <div className="border-b p-4">
          <h2 className="text-lg font-semibold flex items-center gap-2">
            <ArrowLeftRight size={20} />
            Transformation Validation
          </h2>
          <p className="text-sm text-gray-500">
            Comparing expected output (SSIS simulation) vs actual output (Databricks)
          </p>
        </div>
        <div className="p-8 text-center">
          <FlaskConical size={40} className="mx-auto text-gray-400 mb-4" />
          <h3 className="text-lg font-medium text-gray-900 mb-2">No Comparison Data Yet</h3>
          <p className="text-sm text-gray-500 max-w-md mx-auto">
            {comparison?.summary || 'Run the Validation Pipeline above to generate comparison data. The pipeline will simulate the original SSIS output and compare it against the Databricks conversion.'}
          </p>
        </div>
      </div>
    )
  }

  const getMatchIcon = (match: boolean) => {
    return match
      ? <CheckCircle2 className="text-green-600" size={16} />
      : <XCircle className="text-red-600" size={16} />
  }

  return (
    <div className="bg-white rounded-xl shadow-sm border">
      {/* Header */}
      <div className="border-b p-4">
        <div className="flex items-center justify-between">
          <div>
            <h2 className="text-lg font-semibold flex items-center gap-2">
              <ArrowLeftRight size={20} />
              Transformation Validation
            </h2>
            <p className="text-sm text-gray-500">
              Comparing expected output (SSIS simulation) vs actual output (Databricks) — same schema, apples-to-apples
            </p>
          </div>
          {onRefresh && (
            <button
              onClick={onRefresh}
              disabled={isLoading}
              className="px-3 py-1.5 text-sm border rounded-lg hover:bg-gray-50 flex items-center gap-2 disabled:opacity-50"
            >
              <RefreshCw size={14} className={isLoading ? 'animate-spin' : ''} />
              Refresh
            </button>
          )}
        </div>

        {/* Comparison Summary */}
        {comparison && (
          <div className="mt-4 flex gap-4 flex-wrap">
            <div className="flex items-center gap-2 text-sm">
              {getMatchIcon(comparison.row_count_match)}
              <span>Row Count: {comparison.row_count_match ? 'Match' : 'Mismatch'}</span>
            </div>
            <div className="flex items-center gap-2 text-sm">
              {getMatchIcon(comparison.schema_match)}
              <span>Schema: {comparison.schema_match ? 'Match' : 'Mismatch'}</span>
            </div>
            <div className="flex items-center gap-2 text-sm">
              {comparison.sample_match_rate >= 100
                ? <CheckCircle2 className="text-green-600" size={16} />
                : comparison.sample_match_rate >= 95
                ? <AlertTriangle className="text-yellow-600" size={16} />
                : <XCircle className="text-red-600" size={16} />}
              <span>Data Match: {comparison.sample_match_rate.toFixed(1)}%</span>
            </div>
          </div>
        )}

        {/* Summary Message */}
        {comparison && (
          <div className={`mt-3 p-3 rounded-lg text-sm ${
            comparison.sample_match_rate >= 100
              ? 'bg-green-50 text-green-800 border border-green-200'
              : comparison.sample_match_rate >= 95
              ? 'bg-yellow-50 text-yellow-800 border border-yellow-200'
              : 'bg-red-50 text-red-800 border border-red-200'
          }`}>
            {comparison.summary}
          </div>
        )}
      </div>

      {/* Tabs */}
      <div className="border-b flex">
        <button
          onClick={() => setActiveTab('sample')}
          className={`px-4 py-2 text-sm font-medium border-b-2 ${
            activeTab === 'sample'
              ? 'border-[#ff6600] text-[#ff6600]'
              : 'border-transparent text-gray-500 hover:text-gray-700'
          }`}
        >
          Sample Data
        </button>
        <button
          onClick={() => setActiveTab('schema')}
          className={`px-4 py-2 text-sm font-medium border-b-2 ${
            activeTab === 'schema'
              ? 'border-[#ff6600] text-[#ff6600]'
              : 'border-transparent text-gray-500 hover:text-gray-700'
          }`}
        >
          Schema
        </button>
        <button
          onClick={() => setActiveTab('diff')}
          className={`px-4 py-2 text-sm font-medium border-b-2 ${
            activeTab === 'diff'
              ? 'border-[#ff6600] text-[#ff6600]'
              : 'border-transparent text-gray-500 hover:text-gray-700'
          }`}
        >
          Differences
        </button>
      </div>

      {/* Side-by-side panels */}
      <div className="grid grid-cols-1 lg:grid-cols-2 divide-y lg:divide-y-0 lg:divide-x">
        {/* Expected Output Panel (SSIS Simulation) */}
        <div className="flex flex-col">
          <div className="p-3 bg-blue-50 border-b">
            <div className="flex items-center gap-2 font-medium text-blue-900">
              <FlaskConical size={16} />
              Expected Output
              <span className="text-xs font-normal text-blue-600 ml-1">(SSIS Simulation)</span>
            </div>
            {expected && (
              <div className="text-xs text-blue-700 mt-1">
                {expected.catalog ? `${expected.catalog}.${expected.schema}.${expected.table}` : expected.location}
              </div>
            )}
          </div>

          {expected ? (
            <div className="flex-1">
              {/* Stats */}
              <div className="p-3 border-b bg-gray-50 text-sm flex gap-4">
                <span><strong>{expected.row_count.toLocaleString()}</strong> rows</span>
                <span><strong>{expected.column_count}</strong> columns</span>
              </div>

              {/* Content based on tab */}
              {activeTab === 'sample' && (
                <div className="overflow-auto max-h-[400px]">
                  <table className="w-full text-xs">
                    <thead className="bg-gray-100 sticky top-0">
                      <tr>
                        {expected.sample.columns.map((col, i) => (
                          <th key={i} className="px-2 py-1 text-left font-medium whitespace-nowrap border-r last:border-r-0">
                            {col}
                          </th>
                        ))}
                      </tr>
                    </thead>
                    <tbody>
                      {expected.sample.rows.map((row, i) => (
                        <tr key={i} className="border-t hover:bg-gray-50">
                          {row.map((cell, j) => (
                            <td key={j} className="px-2 py-1 whitespace-nowrap border-r last:border-r-0">
                              {cell === null ? <span className="text-gray-400">NULL</span> : String(cell)}
                            </td>
                          ))}
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              )}

              {activeTab === 'schema' && (
                <div className="p-3 space-y-1">
                  {expected.columns.map((col, i) => (
                    <div key={i} className="flex justify-between text-sm py-1 border-b last:border-b-0">
                      <span className="font-mono">{col.name}</span>
                      <span className="text-gray-500">{col.type}</span>
                    </div>
                  ))}
                </div>
              )}

              {activeTab === 'diff' && comparison?.mismatched_columns && (
                <div className="p-3">
                  {comparison.mismatched_columns.length === 0 ? (
                    <div className="text-green-600 text-sm flex items-center gap-2">
                      <CheckCircle2 size={16} />
                      All values match perfectly
                    </div>
                  ) : (
                    <div className="space-y-2">
                      <div className="text-sm font-medium text-red-900 mb-2">
                        {comparison.mismatched_columns.length} difference(s) found:
                      </div>
                      {comparison.mismatched_columns.map((diff, i) => (
                        <div key={i} className="text-sm text-red-600 flex items-center gap-2 bg-red-50 p-2 rounded">
                          <XCircle size={14} className="flex-shrink-0" />
                          {diff}
                        </div>
                      ))}
                    </div>
                  )}
                </div>
              )}
            </div>
          ) : (
            <div className="p-8 text-center text-gray-500">
              <Table2 size={32} className="mx-auto mb-2 opacity-50" />
              No expected data configured
            </div>
          )}
        </div>

        {/* Actual Output Panel (Databricks) */}
        <div className="flex flex-col">
          <div className="p-3 bg-green-50 border-b">
            <div className="flex items-center gap-2 font-medium text-green-900">
              <Cpu size={16} />
              Actual Output
              <span className="text-xs font-normal text-green-600 ml-1">(Databricks)</span>
            </div>
            {actual && (
              <div className="text-xs text-green-700 mt-1">
                {actual.catalog ? `${actual.catalog}.${actual.schema}.${actual.table}` : actual.location}
              </div>
            )}
          </div>

          {actual ? (
            <div className="flex-1">
              {/* Stats */}
              <div className="p-3 border-b bg-gray-50 text-sm flex gap-4">
                <span><strong>{actual.row_count.toLocaleString()}</strong> rows</span>
                <span><strong>{actual.column_count}</strong> columns</span>
              </div>

              {/* Content based on tab */}
              {activeTab === 'sample' && (
                <div className="overflow-auto max-h-[400px]">
                  <table className="w-full text-xs">
                    <thead className="bg-gray-100 sticky top-0">
                      <tr>
                        {actual.sample.columns.map((col, i) => (
                          <th key={i} className="px-2 py-1 text-left font-medium whitespace-nowrap border-r last:border-r-0">
                            {col}
                          </th>
                        ))}
                      </tr>
                    </thead>
                    <tbody>
                      {actual.sample.rows.map((row, i) => (
                        <tr key={i} className="border-t hover:bg-gray-50">
                          {row.map((cell, j) => (
                            <td key={j} className="px-2 py-1 whitespace-nowrap border-r last:border-r-0">
                              {cell === null ? <span className="text-gray-400">NULL</span> : String(cell)}
                            </td>
                          ))}
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              )}

              {activeTab === 'schema' && (
                <div className="p-3 space-y-1">
                  {actual.columns.map((col, i) => (
                    <div key={i} className="flex justify-between text-sm py-1 border-b last:border-b-0">
                      <span className="font-mono">{col.name}</span>
                      <span className="text-gray-500">{col.type}</span>
                    </div>
                  ))}
                </div>
              )}

              {activeTab === 'diff' && comparison && (
                <div className="p-3">
                  <div className="text-sm text-gray-600">
                    {comparison.sample_match_rate >= 100 ? (
                      <div className="flex items-center gap-2 text-green-600">
                        <CheckCircle2 size={16} />
                        Databricks output matches expected SSIS output perfectly
                      </div>
                    ) : (
                      <div className="text-gray-700">
                        Review the differences on the left panel to understand where the Databricks output differs from expected.
                      </div>
                    )}
                  </div>
                </div>
              )}
            </div>
          ) : (
            <div className="p-8 text-center text-gray-500">
              <Table2 size={32} className="mx-auto mb-2 opacity-50" />
              No output data yet
            </div>
          )}
        </div>
      </div>
    </div>
  )
}

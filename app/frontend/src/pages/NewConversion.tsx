import { useState, useCallback } from 'react'
import { useNavigate } from 'react-router-dom'
import { useMutation, useQuery } from '@tanstack/react-query'
import {
  Upload, FileCode2, X, Cpu, Zap, Brain, FileText, Plus, Trash2,
  ChevronDown, ChevronUp, Coins, Info, FolderGit2, Database,
  HardDrive, Globe, Sparkles, AlertCircle, Loader2
} from 'lucide-react'
import axios from 'axios'
import FileBrowser from '../components/FileBrowser'

interface UploadResponse {
  job_id: string
  file_path: string
  file_name: string
  file_size: number
}

interface AIModel {
  id: string
  name: string
  tier: string
  description: string
  cost: string
}

interface ReferenceFile {
  file_id: string
  file_name: string
  file_type: string
}

type SourceTab = 'upload' | 'workspace' | 'volume' | 'repo'
type OutputFormat = 'pyspark' | 'dlt_sdp' | 'dbt'
type SourceFormat = 'auto' | 'ssis' | 'sql_script' | 'stored_proc' | 'informatica_pc'
type ValidationSourceType = 'none' | 'uc_table' | 'federated' | 'synthetic'

const SOURCE_TABS: { key: SourceTab; label: string; icon: React.ReactNode }[] = [
  { key: 'upload', label: 'Upload', icon: <Upload size={16} /> },
  { key: 'workspace', label: 'Workspace', icon: <FolderGit2 size={16} /> },
  { key: 'volume', label: 'UC Volume', icon: <Database size={16} /> },
  { key: 'repo', label: 'Repo / Git', icon: <Globe size={16} /> },
]

const OUTPUT_FORMATS: { key: OutputFormat; label: string; description: string }[] = [
  { key: 'pyspark', label: 'PySpark Notebooks', description: 'Standard Databricks notebooks with PySpark transformations' },
  { key: 'dlt_sdp', label: 'Spark Declarative Pipelines', description: 'DLT/SDP with CREATE OR REFRESH, expectations, and streaming tables' },
  { key: 'dbt', label: 'dbt Models', description: 'dbt SQL models with ref(), source(), and schema.yml' },
]

const SOURCE_FORMATS: { key: SourceFormat; label: string }[] = [
  { key: 'auto', label: 'Auto-Detect' },
  { key: 'ssis', label: 'SSIS Package' },
  { key: 'sql_script', label: 'SQL Server Script' },
  { key: 'stored_proc', label: 'Stored Procedures' },
  { key: 'informatica_pc', label: 'Informatica PowerCenter' },
]

export default function NewConversion() {
  const navigate = useNavigate()

  // Source code
  const [sourceTab, setSourceTab] = useState<SourceTab>('upload')
  const [file, setFile] = useState<File | null>(null)
  const [isDragging, setIsDragging] = useState(false)
  const [selectedRemotePath, setSelectedRemotePath] = useState<string | null>(null)
  const [selectedRemoteName, setSelectedRemoteName] = useState<string | null>(null)
  const [workspacePath, setWorkspacePath] = useState('/Workspace/Users/')
  const [volumePath, setVolumePath] = useState('/Volumes/')
  const [repoPath, setRepoPath] = useState('/Repos/')

  // Format
  const [sourceFormat, setSourceFormat] = useState<SourceFormat>('auto')
  const [detectedFormat, setDetectedFormat] = useState<string | null>(null)
  const [outputFormat, setOutputFormat] = useState<OutputFormat>('pyspark')

  // Validation
  const [showValidation, setShowValidation] = useState(false)
  const [validationSource, setValidationSource] = useState<ValidationSourceType>('none')
  const [validationTable, setValidationTable] = useState('')

  // Model & instructions
  const [selectedModel, setSelectedModel] = useState('databricks-claude-haiku-4-5')
  const [conversionInstructions, setConversionInstructions] = useState('')
  const [showContext, setShowContext] = useState(false)
  const [referenceFiles, setReferenceFiles] = useState<ReferenceFile[]>([])
  const [uploadingRef, setUploadingRef] = useState(false)

  // Job
  const [jobName, setJobName] = useState('')
  const [autoRun, setAutoRun] = useState(true)

  // Fetch available models
  const { data: models = [] } = useQuery<AIModel[]>({
    queryKey: ['models'],
    queryFn: async () => {
      const { data } = await axios.get('/api/conversions/models')
      return data
    },
  })

  // Auto-detect format when a file is selected
  const detectFormat = async (filename: string, path?: string) => {
    try {
      const { data } = await axios.post('/api/files/detect-format', { filename, path })
      setDetectedFormat(data.detected_format)
    } catch {
      // Fallback: infer from extension
      if (filename.endsWith('.dtsx')) setDetectedFormat('ssis')
      else if (filename.endsWith('.xml')) setDetectedFormat('informatica_pc')
      else setDetectedFormat('sql_script')
    }
  }

  // Upload mutation
  const uploadMutation = useMutation({
    mutationFn: async () => {
      let sourcePath: string
      let fileName: string
      let sourceLocation: string

      if (sourceTab === 'upload') {
        if (!file) throw new Error('No file selected')

        const formData = new FormData()
        formData.append('file', file)

        const { data: uploadResult } = await axios.post<UploadResponse>(
          '/api/files/upload',
          formData,
          { headers: { 'Content-Type': 'multipart/form-data' } }
        )
        sourcePath = uploadResult.file_path
        fileName = file.name
        sourceLocation = 'upload'
      } else {
        // Remote file: ingest it into Volumes first
        if (!selectedRemotePath || !selectedRemoteName) throw new Error('No file selected')

        const sourceType = sourceTab === 'workspace' ? 'workspace' : 'volume'
        const { data: ingestResult } = await axios.post<UploadResponse>(
          '/api/files/ingest-remote',
          {
            source_path: selectedRemotePath,
            source_type: sourceType,
          }
        )
        sourcePath = ingestResult.file_path
        fileName = selectedRemoteName
        sourceLocation = sourceTab === 'workspace' ? 'workspace'
          : sourceTab === 'volume' ? 'uc_volume'
          : 'repo'
      }

      // Determine source type
      const resolvedFormat = sourceFormat === 'auto'
        ? (detectedFormat || 'sql_script')
        : sourceFormat

      // Create conversion job
      const { data: job } = await axios.post('/api/conversions', {
        job_name: jobName || fileName,
        source_type: resolvedFormat,
        source_path: sourcePath,
        output_format: outputFormat,
        source_location: sourceLocation,
        ai_model: selectedModel,
        conversion_instructions: conversionInstructions || undefined,
        reference_file_ids: referenceFiles.length > 0 ? referenceFiles.map(f => f.file_id) : undefined,
        validation_source: validationSource !== 'none' ? validationSource : undefined,
        validation_table: validationTable || undefined,
      })

      // Auto-run conversion if enabled
      if (autoRun) {
        await axios.post(`/api/conversions/${job.job_id}/run`)
      }

      return job
    },
    onSuccess: (job) => {
      navigate(`/conversions/${job.job_id}`)
    },
  })

  // File handlers
  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault()
    setIsDragging(false)

    const droppedFile = e.dataTransfer.files[0]
    if (droppedFile) {
      const ext = droppedFile.name.split('.').pop()?.toLowerCase()
      if (['dtsx', 'sql', 'xml', 'txt'].includes(ext || '')) {
        setFile(droppedFile)
        if (!jobName) setJobName(droppedFile.name.replace(/\.[^/.]+$/, ''))
        detectFormat(droppedFile.name)
      }
    }
  }, [jobName])

  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    const selectedFile = e.target.files?.[0]
    if (selectedFile) {
      setFile(selectedFile)
      if (!jobName) setJobName(selectedFile.name.replace(/\.[^/.]+$/, ''))
      detectFormat(selectedFile.name)
    }
  }

  const handleRemoteSelect = (path: string, name: string) => {
    setSelectedRemotePath(path)
    setSelectedRemoteName(name)
    if (!jobName) setJobName(name.replace(/\.[^/.]+$/, ''))
    detectFormat(name, path)
  }

  const handleReferenceUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const refFile = e.target.files?.[0]
    if (!refFile) return

    setUploadingRef(true)
    try {
      const formData = new FormData()
      formData.append('file', refFile)

      const { data } = await axios.post('/api/files/reference/upload', formData, {
        headers: { 'Content-Type': 'multipart/form-data' }
      })

      setReferenceFiles([...referenceFiles, {
        file_id: data.file_id,
        file_name: data.file_name,
        file_type: data.file_type,
      }])
    } catch (error) {
      console.error('Failed to upload reference file:', error)
    } finally {
      setUploadingRef(false)
    }
  }

  const removeReferenceFile = async (fileId: string) => {
    try {
      await axios.delete(`/api/files/reference/${fileId}`)
      setReferenceFiles(referenceFiles.filter(f => f.file_id !== fileId))
    } catch (error) {
      console.error('Failed to delete reference file:', error)
    }
  }

  // Token estimation
  const estimateTokens = (fileSize: number, isXml: boolean) => {
    const charsPerToken = isXml ? 3.5 : 4
    const estimatedInputTokens = Math.ceil(fileSize / charsPerToken)
    const estimatedOutputTokens = Math.ceil(estimatedInputTokens * 2.5)
    return { input: estimatedInputTokens, output: estimatedOutputTokens, total: estimatedInputTokens + estimatedOutputTokens }
  }

  const modelCosts: Record<string, { input: number; output: number }> = {
    'databricks-claude-haiku-4-5': { input: 1.0, output: 5.0 },
    'databricks-gpt-oss-120b': { input: 0.5, output: 2.0 },
    'databricks-gpt-oss-20b': { input: 0.2, output: 0.8 },
    'databricks-gpt-5-nano': { input: 0.3, output: 1.0 },
    'databricks-claude-opus-4-5': { input: 15.0, output: 75.0 },
  }

  const tokenEstimate = file ? estimateTokens(file.size, file.name.endsWith('.dtsx') || file.name.endsWith('.xml')) : null
  const costs = modelCosts[selectedModel] || { input: 1.0, output: 5.0 }
  const estimatedCost = tokenEstimate
    ? ((tokenEstimate.input / 1_000_000) * costs.input) + ((tokenEstimate.output / 1_000_000) * costs.output)
    : 0

  const hasSelection = sourceTab === 'upload' ? !!file : !!selectedRemotePath

  const effectiveFormat = sourceFormat === 'auto' ? detectedFormat : sourceFormat

  return (
    <div className="p-8 max-w-4xl mx-auto">
      <div className="mb-8">
        <h1 className="text-2xl font-bold text-gray-900">New Conversion</h1>
        <p className="text-gray-600">Configure and run a source code conversion to Databricks</p>
      </div>

      {/* ============ Section 1: Source Code ============ */}
      <section className="mb-8">
        <h2 className="text-lg font-semibold text-gray-900 mb-4 flex items-center gap-2">
          <FileCode2 size={20} className="text-[#ff6600]" />
          Source Code
        </h2>

        {/* Tabs */}
        <div className="flex border-b mb-4">
          {SOURCE_TABS.map((tab) => (
            <button
              key={tab.key}
              onClick={() => {
                setSourceTab(tab.key)
                setSelectedRemotePath(null)
                setSelectedRemoteName(null)
              }}
              className={`flex items-center gap-2 px-4 py-2.5 text-sm font-medium border-b-2 transition-colors ${
                sourceTab === tab.key
                  ? 'border-[#ff6600] text-[#ff6600]'
                  : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
              }`}
            >
              {tab.icon}
              {tab.label}
            </button>
          ))}
        </div>

        {/* Upload Tab */}
        {sourceTab === 'upload' && (
          <div
            onDragOver={(e) => { e.preventDefault(); setIsDragging(true) }}
            onDragLeave={() => setIsDragging(false)}
            onDrop={handleDrop}
            className={`border-2 border-dashed rounded-xl p-8 text-center transition-colors ${
              isDragging ? 'border-[#ff6600] bg-orange-50' : 'border-gray-300 hover:border-gray-400'
            }`}
          >
            {file ? (
              <div className="flex items-center justify-between bg-gray-50 rounded-lg p-4">
                <div className="flex items-center gap-3">
                  <FileCode2 className="text-[#ff6600]" size={24} />
                  <div className="text-left">
                    <div className="font-medium">{file.name}</div>
                    <div className="text-sm text-gray-500">{(file.size / 1024).toFixed(1)} KB</div>
                  </div>
                  {detectedFormat && sourceFormat === 'auto' && (
                    <span className="text-xs px-2 py-1 bg-blue-100 text-blue-700 rounded-full">
                      {detectedFormat.replace('_', ' ')}
                    </span>
                  )}
                </div>
                <button
                  onClick={() => { setFile(null); setDetectedFormat(null) }}
                  className="p-1 hover:bg-gray-200 rounded"
                >
                  <X size={20} />
                </button>
              </div>
            ) : (
              <>
                <Upload className="mx-auto text-gray-400 mb-4" size={48} />
                <p className="text-gray-600 mb-2">Drag and drop your file here, or</p>
                <label className="inline-flex px-4 py-2 bg-white border rounded-lg cursor-pointer hover:bg-gray-50">
                  Browse Files
                  <input
                    type="file"
                    accept=".dtsx,.sql,.xml,.txt"
                    onChange={handleFileSelect}
                    className="hidden"
                  />
                </label>
                <p className="text-sm text-gray-500 mt-4">Supported: .dtsx (SSIS), .sql, .xml (Informatica), .txt</p>
              </>
            )}
          </div>
        )}

        {/* Workspace Tab */}
        {sourceTab === 'workspace' && (
          <div>
            <div className="flex gap-2 mb-3">
              <input
                type="text"
                value={workspacePath}
                onChange={(e) => setWorkspacePath(e.target.value)}
                placeholder="/Workspace/Users/your.name@company.com/"
                className="flex-1 px-3 py-2 border rounded-lg text-sm focus:ring-2 focus:ring-[#ff6600] focus:border-transparent"
              />
              <button
                onClick={() => setWorkspacePath(workspacePath)}
                className="px-4 py-2 bg-gray-100 border rounded-lg text-sm hover:bg-gray-200"
              >
                Browse
              </button>
            </div>
            <FileBrowser
              source="workspace"
              rootPath={workspacePath}
              onSelect={handleRemoteSelect}
            />
            {selectedRemoteName && (
              <div className="mt-3 flex items-center gap-2 p-3 bg-orange-50 border border-[#ff6600] rounded-lg">
                <FileCode2 size={16} className="text-[#ff6600]" />
                <span className="text-sm font-medium">{selectedRemoteName}</span>
                {detectedFormat && sourceFormat === 'auto' && (
                  <span className="text-xs px-2 py-0.5 bg-blue-100 text-blue-700 rounded-full">
                    {detectedFormat.replace('_', ' ')}
                  </span>
                )}
              </div>
            )}
          </div>
        )}

        {/* UC Volume Tab */}
        {sourceTab === 'volume' && (
          <div>
            <div className="flex gap-2 mb-3">
              <input
                type="text"
                value={volumePath}
                onChange={(e) => setVolumePath(e.target.value)}
                placeholder="/Volumes/catalog/schema/volume/"
                className="flex-1 px-3 py-2 border rounded-lg text-sm focus:ring-2 focus:ring-[#ff6600] focus:border-transparent"
              />
              <button
                onClick={() => setVolumePath(volumePath)}
                className="px-4 py-2 bg-gray-100 border rounded-lg text-sm hover:bg-gray-200"
              >
                Browse
              </button>
            </div>
            <FileBrowser
              source="volume"
              rootPath={volumePath}
              onSelect={handleRemoteSelect}
            />
            {selectedRemoteName && (
              <div className="mt-3 flex items-center gap-2 p-3 bg-orange-50 border border-[#ff6600] rounded-lg">
                <FileCode2 size={16} className="text-[#ff6600]" />
                <span className="text-sm font-medium">{selectedRemoteName}</span>
                {detectedFormat && sourceFormat === 'auto' && (
                  <span className="text-xs px-2 py-0.5 bg-blue-100 text-blue-700 rounded-full">
                    {detectedFormat.replace('_', ' ')}
                  </span>
                )}
              </div>
            )}
          </div>
        )}

        {/* Repo Tab */}
        {sourceTab === 'repo' && (
          <div>
            <div className="flex gap-2 mb-3">
              <input
                type="text"
                value={repoPath}
                onChange={(e) => setRepoPath(e.target.value)}
                placeholder="/Repos/your.name@company.com/repo-name"
                className="flex-1 px-3 py-2 border rounded-lg text-sm focus:ring-2 focus:ring-[#ff6600] focus:border-transparent"
              />
              <button
                onClick={() => setRepoPath(repoPath)}
                className="px-4 py-2 bg-gray-100 border rounded-lg text-sm hover:bg-gray-200"
              >
                Browse
              </button>
            </div>
            <FileBrowser
              source="repo"
              rootPath={repoPath}
              onSelect={handleRemoteSelect}
            />
            {selectedRemoteName && (
              <div className="mt-3 flex items-center gap-2 p-3 bg-orange-50 border border-[#ff6600] rounded-lg">
                <FileCode2 size={16} className="text-[#ff6600]" />
                <span className="text-sm font-medium">{selectedRemoteName}</span>
                {detectedFormat && sourceFormat === 'auto' && (
                  <span className="text-xs px-2 py-0.5 bg-blue-100 text-blue-700 rounded-full">
                    {detectedFormat.replace('_', ' ')}
                  </span>
                )}
              </div>
            )}
          </div>
        )}
      </section>

      {/* Token Estimation Card (upload only) */}
      {sourceTab === 'upload' && file && tokenEstimate && (
        <div className="mb-6 p-4 bg-blue-50 border border-blue-200 rounded-lg">
          <div className="flex items-start gap-3">
            <Coins className="text-blue-600 mt-0.5" size={20} />
            <div className="flex-1">
              <h3 className="font-medium text-blue-900 mb-2">Estimated Token Usage</h3>
              <div className="grid grid-cols-3 gap-4 text-sm">
                <div>
                  <div className="text-blue-700 font-medium">{tokenEstimate.input.toLocaleString()}</div>
                  <div className="text-blue-600 text-xs">Input tokens</div>
                </div>
                <div>
                  <div className="text-blue-700 font-medium">{tokenEstimate.output.toLocaleString()}</div>
                  <div className="text-blue-600 text-xs">Output tokens (est)</div>
                </div>
                <div>
                  <div className="text-blue-700 font-medium">${estimatedCost.toFixed(4)}</div>
                  <div className="text-blue-600 text-xs">Est. cost</div>
                </div>
              </div>
              <div className="mt-2 flex items-center gap-1 text-xs text-blue-600">
                <Info size={12} />
                Based on file size and selected model pricing
              </div>
            </div>
          </div>
        </div>
      )}

      {/* ============ Section 2: Source Format & Output Format ============ */}
      <section className="mb-8">
        <h2 className="text-lg font-semibold text-gray-900 mb-4 flex items-center gap-2">
          <Sparkles size={20} className="text-[#ff6600]" />
          Format Configuration
        </h2>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          {/* Source Format */}
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Source Format
            </label>
            <select
              value={sourceFormat}
              onChange={(e) => setSourceFormat(e.target.value as SourceFormat)}
              className="w-full px-3 py-2 border rounded-lg text-sm focus:ring-2 focus:ring-[#ff6600] focus:border-transparent"
            >
              {SOURCE_FORMATS.map((fmt) => (
                <option key={fmt.key} value={fmt.key}>
                  {fmt.label}
                  {fmt.key === 'auto' && detectedFormat ? ` → ${detectedFormat.replace('_', ' ')}` : ''}
                </option>
              ))}
            </select>
            {sourceFormat === 'auto' && detectedFormat && (
              <p className="text-xs text-green-600 mt-1 flex items-center gap-1">
                <Sparkles size={12} />
                Detected: {detectedFormat.replace('_', ' ')}
              </p>
            )}
            {sourceFormat === 'auto' && !detectedFormat && hasSelection && (
              <p className="text-xs text-gray-500 mt-1">Select a file to auto-detect format</p>
            )}
          </div>

          {/* Output Format */}
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Output Format
            </label>
            <div className="space-y-2">
              {OUTPUT_FORMATS.map((fmt) => (
                <label
                  key={fmt.key}
                  className={`flex items-start gap-3 p-3 border rounded-lg cursor-pointer transition-colors ${
                    outputFormat === fmt.key
                      ? 'border-[#ff6600] bg-orange-50'
                      : 'border-gray-200 hover:border-gray-300'
                  }`}
                >
                  <input
                    type="radio"
                    name="outputFormat"
                    value={fmt.key}
                    checked={outputFormat === fmt.key}
                    onChange={(e) => setOutputFormat(e.target.value as OutputFormat)}
                    className="sr-only"
                  />
                  <div className={`mt-0.5 w-4 h-4 rounded-full border-2 flex items-center justify-center ${
                    outputFormat === fmt.key ? 'border-[#ff6600]' : 'border-gray-300'
                  }`}>
                    {outputFormat === fmt.key && <div className="w-2 h-2 rounded-full bg-[#ff6600]" />}
                  </div>
                  <div>
                    <div className="font-medium text-sm">{fmt.label}</div>
                    <div className="text-xs text-gray-500">{fmt.description}</div>
                  </div>
                </label>
              ))}
            </div>
          </div>
        </div>
      </section>

      {/* ============ Section 3: Validation Source (collapsible) ============ */}
      <section className="mb-8">
        <button
          onClick={() => setShowValidation(!showValidation)}
          className="flex items-center gap-2 text-lg font-semibold text-gray-900 hover:text-gray-700"
        >
          {showValidation ? <ChevronUp size={20} /> : <ChevronDown size={20} />}
          <HardDrive size={20} className="text-[#ff6600]" />
          Validation Source
          <span className="text-sm font-normal text-gray-500">(optional)</span>
        </button>

        {showValidation && (
          <div className="mt-4 p-4 bg-gray-50 rounded-lg">
            <p className="text-sm text-gray-600 mb-4">
              Configure how to validate the converted output against source data
            </p>
            <div className="space-y-3">
              {[
                { key: 'none', label: 'Skip Validation', desc: 'No automated data comparison' },
                { key: 'uc_table', label: 'Unity Catalog Table', desc: 'Compare against a UC table (e.g. replicated via Lakeflow Connect)' },
                { key: 'federated', label: 'Federated Table', desc: 'Compare against a federated query table (e.g. SQL Server, Oracle via Lakehouse Federation)' },
                { key: 'synthetic', label: 'Generate Synthetic', desc: 'Auto-generate test data from the extracted source schema using Faker' },
              ].map((opt) => (
                <label
                  key={opt.key}
                  className={`flex items-center gap-3 p-3 border rounded-lg cursor-pointer ${
                    validationSource === opt.key
                      ? 'border-[#ff6600] bg-orange-50'
                      : 'border-gray-200 hover:border-gray-300 bg-white'
                  }`}
                >
                  <input
                    type="radio"
                    name="validationSource"
                    value={opt.key}
                    checked={validationSource === opt.key}
                    onChange={(e) => setValidationSource(e.target.value as ValidationSourceType)}
                    className="sr-only"
                  />
                  <div className={`w-4 h-4 rounded-full border-2 flex items-center justify-center ${
                    validationSource === opt.key ? 'border-[#ff6600]' : 'border-gray-300'
                  }`}>
                    {validationSource === opt.key && <div className="w-2 h-2 rounded-full bg-[#ff6600]" />}
                  </div>
                  <div>
                    <div className="font-medium text-sm">{opt.label}</div>
                    <div className="text-xs text-gray-500">{opt.desc}</div>
                  </div>
                </label>
              ))}
            </div>

            {(validationSource === 'uc_table' || validationSource === 'federated') && (
              <div className="mt-4">
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  Source Table (3-level name)
                </label>
                <input
                  type="text"
                  value={validationTable}
                  onChange={(e) => setValidationTable(e.target.value)}
                  placeholder="catalog.schema.table_name"
                  className="w-full px-3 py-2 border rounded-lg text-sm focus:ring-2 focus:ring-[#ff6600] focus:border-transparent"
                />
                <p className="text-xs text-gray-500 mt-1">
                  {validationSource === 'uc_table'
                    ? 'The UC table containing real source data. The simulator will use this data instead of generating synthetic rows.'
                    : 'A federated table connected via Lakehouse Federation. Data will be sampled for comparison.'}
                </p>
              </div>
            )}
          </div>
        )}
      </section>

      {/* ============ Section 4: AI Model ============ */}
      <section className="mb-8">
        <h2 className="text-lg font-semibold text-gray-900 mb-4 flex items-center gap-2">
          <Brain size={20} className="text-[#ff6600]" />
          AI Model
        </h2>
        <div className="grid gap-3">
          {models.map((model) => (
            <label
              key={model.id}
              className={`flex items-center gap-4 p-4 border rounded-lg cursor-pointer transition-colors ${
                selectedModel === model.id
                  ? 'border-[#ff6600] bg-orange-50'
                  : 'border-gray-200 hover:border-gray-300'
              }`}
            >
              <input
                type="radio"
                name="model"
                value={model.id}
                checked={selectedModel === model.id}
                onChange={(e) => setSelectedModel(e.target.value)}
                className="sr-only"
              />
              <div className={`p-2 rounded-lg ${
                model.tier === 'fast' ? 'bg-green-100 text-green-600' :
                model.tier === 'balanced' ? 'bg-blue-100 text-blue-600' :
                'bg-purple-100 text-purple-600'
              }`}>
                {model.tier === 'fast' ? <Zap size={20} /> :
                 model.tier === 'balanced' ? <Cpu size={20} /> :
                 <Brain size={20} />}
              </div>
              <div className="flex-1">
                <div className="flex items-center gap-2">
                  <span className="font-medium">{model.name}</span>
                  <span className="text-xs px-2 py-0.5 bg-gray-100 rounded-full">
                    {model.cost}
                  </span>
                </div>
                <p className="text-sm text-gray-500">{model.description}</p>
              </div>
              {selectedModel === model.id && (
                <div className="w-2 h-2 bg-[#ff6600] rounded-full"></div>
              )}
            </label>
          ))}
        </div>
      </section>

      {/* ============ Section 5: Instructions & Context (collapsible) ============ */}
      <section className="mb-8">
        <button
          onClick={() => setShowContext(!showContext)}
          className="flex items-center gap-2 text-lg font-semibold text-gray-900 hover:text-gray-700"
        >
          {showContext ? <ChevronUp size={20} /> : <ChevronDown size={20} />}
          <FileText size={20} className="text-[#ff6600]" />
          Instructions & Context
          <span className="text-sm font-normal text-gray-500">(optional)</span>
        </button>

        {showContext && (
          <div className="mt-4 space-y-6 p-4 bg-gray-50 rounded-lg">
            {/* Conversion Instructions */}
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Conversion Instructions
              </label>
              <textarea
                value={conversionInstructions}
                onChange={(e) => setConversionInstructions(e.target.value)}
                placeholder="Add specific instructions for the conversion (e.g., 'Use Unity Catalog table catalog.schema.table for the destination', 'Implement error handling with try/except blocks')"
                rows={4}
                className="w-full px-4 py-2 border rounded-lg focus:ring-2 focus:ring-[#ff6600] focus:border-transparent"
              />
              <p className="text-xs text-gray-500 mt-1">
                These instructions will be passed to the AI model to guide the conversion
              </p>
            </div>

            {/* Reference Files */}
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Reference Files
              </label>
              <p className="text-xs text-gray-500 mb-3">
                Upload documentation, example code, or style guides to help inform the conversion
              </p>

              {referenceFiles.length > 0 && (
                <div className="mb-3 space-y-2">
                  {referenceFiles.map((ref) => (
                    <div
                      key={ref.file_id}
                      className="flex items-center justify-between bg-white p-3 rounded-lg border"
                    >
                      <div className="flex items-center gap-2">
                        <FileText size={16} className="text-gray-500" />
                        <span className="text-sm">{ref.file_name}</span>
                        <span className="text-xs px-2 py-0.5 bg-gray-100 rounded">
                          {ref.file_type}
                        </span>
                      </div>
                      <button
                        onClick={() => removeReferenceFile(ref.file_id)}
                        className="p-1 hover:bg-gray-100 rounded text-red-500"
                      >
                        <Trash2 size={16} />
                      </button>
                    </div>
                  ))}
                </div>
              )}

              <label className={`inline-flex items-center gap-2 px-4 py-2 border rounded-lg cursor-pointer hover:bg-white ${
                uploadingRef ? 'opacity-50 cursor-wait' : ''
              }`}>
                <Plus size={16} />
                {uploadingRef ? 'Uploading...' : 'Add Reference File'}
                <input
                  type="file"
                  accept=".md,.sql,.txt,.py,.json,.yaml,.yml"
                  onChange={handleReferenceUpload}
                  disabled={uploadingRef}
                  className="hidden"
                />
              </label>
              <span className="text-xs text-gray-500 ml-2">
                .md, .sql, .txt, .py, .json, .yaml
              </span>
            </div>
          </div>
        )}
      </section>

      {/* ============ Section 6: Job Name & Submit ============ */}
      <section className="border-t pt-6">
        <div className="mb-4">
          <label className="block text-sm font-medium text-gray-700 mb-2">
            Job Name
          </label>
          <input
            type="text"
            value={jobName}
            onChange={(e) => setJobName(e.target.value)}
            placeholder="Enter a name for this conversion"
            className="w-full px-4 py-2 border rounded-lg focus:ring-2 focus:ring-[#ff6600] focus:border-transparent"
          />
        </div>

        <label className="flex items-center gap-3 cursor-pointer mb-6">
          <input
            type="checkbox"
            checked={autoRun}
            onChange={(e) => setAutoRun(e.target.checked)}
            className="w-5 h-5 rounded border-gray-300 text-[#ff6600] focus:ring-[#ff6600]"
          />
          <div>
            <span className="font-medium">Start conversion immediately</span>
            <p className="text-sm text-gray-500">Automatically run conversion after upload</p>
          </div>
        </label>

        {/* Summary bar */}
        {hasSelection && (
          <div className="mb-4 p-3 bg-gray-50 rounded-lg flex items-center gap-4 text-sm">
            <span className="text-gray-500">Summary:</span>
            <span className="font-medium">
              {sourceTab === 'upload' ? file?.name : selectedRemoteName}
            </span>
            <span className="text-gray-400">→</span>
            <span className="px-2 py-0.5 bg-blue-100 text-blue-700 rounded text-xs">
              {effectiveFormat?.replace('_', ' ') || 'auto'}
            </span>
            <span className="text-gray-400">→</span>
            <span className="px-2 py-0.5 bg-green-100 text-green-700 rounded text-xs">
              {OUTPUT_FORMATS.find(f => f.key === outputFormat)?.label}
            </span>
          </div>
        )}

        <button
          onClick={() => uploadMutation.mutate()}
          disabled={!hasSelection || uploadMutation.isPending}
          className="w-full px-4 py-3 bg-[#ff6600] text-white rounded-lg hover:bg-[#e55c00] disabled:opacity-50 disabled:cursor-not-allowed font-medium flex items-center justify-center gap-2"
        >
          {uploadMutation.isPending ? (
            <>
              <Loader2 size={18} className="animate-spin" />
              {autoRun ? 'Uploading & Starting...' : 'Creating Job...'}
            </>
          ) : (
            autoRun ? 'Upload & Convert' : 'Create Conversion Job'
          )}
        </button>

        {uploadMutation.isError && (
          <div className="mt-4 p-4 bg-red-50 text-red-700 rounded-lg flex items-center gap-2">
            <AlertCircle size={16} />
            Error: {(uploadMutation.error as Error).message}
          </div>
        )}
      </section>
    </div>
  )
}

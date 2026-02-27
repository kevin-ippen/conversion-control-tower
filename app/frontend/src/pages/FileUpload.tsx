import { useState, useCallback } from 'react'
import { useNavigate } from 'react-router-dom'
import { useMutation, useQuery } from '@tanstack/react-query'
import { Upload, FileCode2, X, Cpu, Zap, Brain, FileText, Plus, Trash2, ChevronDown, ChevronUp, Coins, Info } from 'lucide-react'
import axios from 'axios'

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

export default function FileUpload() {
  const navigate = useNavigate()
  const [file, setFile] = useState<File | null>(null)
  const [jobName, setJobName] = useState('')
  const [isDragging, setIsDragging] = useState(false)
  const [selectedModel, setSelectedModel] = useState('databricks-claude-haiku-4-5')
  const [autoRun, setAutoRun] = useState(true)
  const [conversionInstructions, setConversionInstructions] = useState('')
  const [showAdvanced, setShowAdvanced] = useState(false)
  const [referenceFiles, setReferenceFiles] = useState<ReferenceFile[]>([])
  const [uploadingRef, setUploadingRef] = useState(false)

  // Fetch available models
  const { data: models = [] } = useQuery<AIModel[]>({
    queryKey: ['models'],
    queryFn: async () => {
      const { data } = await axios.get('/api/conversions/models')
      return data
    },
  })

  const uploadMutation = useMutation({
    mutationFn: async () => {
      if (!file) throw new Error('No file selected')

      const formData = new FormData()
      formData.append('file', file)

      const { data: uploadResult } = await axios.post<UploadResponse>(
        '/api/files/upload',
        formData,
        { headers: { 'Content-Type': 'multipart/form-data' } }
      )

      // Create conversion job
      const sourceType = file.name.endsWith('.dtsx') ? 'ssis' : 'sql_script'
      const { data: job } = await axios.post('/api/conversions', {
        job_name: jobName || file.name,
        source_type: sourceType,
        source_path: uploadResult.file_path,
        ai_model: selectedModel,
        conversion_instructions: conversionInstructions || undefined,
        reference_file_ids: referenceFiles.map(f => f.file_id),
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

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault()
    setIsDragging(false)

    const droppedFile = e.dataTransfer.files[0]
    if (droppedFile && (droppedFile.name.endsWith('.dtsx') || droppedFile.name.endsWith('.sql'))) {
      setFile(droppedFile)
      if (!jobName) {
        setJobName(droppedFile.name.replace(/\.[^/.]+$/, ''))
      }
    }
  }, [jobName])

  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    const selectedFile = e.target.files?.[0]
    if (selectedFile) {
      setFile(selectedFile)
      if (!jobName) {
        setJobName(selectedFile.name.replace(/\.[^/.]+$/, ''))
      }
    }
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

  // Token estimation helper
  const estimateTokens = (fileSize: number, isXml: boolean) => {
    // Rough estimation: 4 characters per token for code/XML
    // XML files are ~30% more verbose due to tags
    const charsPerToken = isXml ? 3.5 : 4
    const estimatedInputTokens = Math.ceil(fileSize / charsPerToken)
    // Output is typically 2-3x input for conversions
    const estimatedOutputTokens = Math.ceil(estimatedInputTokens * 2.5)
    return { input: estimatedInputTokens, output: estimatedOutputTokens, total: estimatedInputTokens + estimatedOutputTokens }
  }

  // Cost per 1M tokens (approximate)
  const modelCosts: Record<string, { input: number; output: number }> = {
    'databricks-claude-haiku-4-5': { input: 1.0, output: 5.0 },
    'databricks-gpt-oss-120b': { input: 0.5, output: 2.0 },
    'databricks-gpt-oss-20b': { input: 0.2, output: 0.8 },
    'databricks-gpt-5-nano': { input: 0.3, output: 1.0 },
    'databricks-claude-opus-4-5': { input: 15.0, output: 75.0 },
  }

  const tokenEstimate = file ? estimateTokens(file.size, file.name.endsWith('.dtsx')) : null
  const costs = modelCosts[selectedModel] || { input: 1.0, output: 5.0 }
  const estimatedCost = tokenEstimate
    ? ((tokenEstimate.input / 1_000_000) * costs.input) + ((tokenEstimate.output / 1_000_000) * costs.output)
    : 0

  return (
    <div className="p-8 max-w-3xl mx-auto">
      <div className="mb-8">
        <h1 className="text-2xl font-bold text-gray-900">Upload Source File</h1>
        <p className="text-gray-600">Upload an SSIS package (.dtsx) or SQL script to convert</p>
      </div>

      {/* Drop Zone */}
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
            </div>
            <button
              onClick={() => setFile(null)}
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
                accept=".dtsx,.sql"
                onChange={handleFileSelect}
                className="hidden"
              />
            </label>
            <p className="text-sm text-gray-500 mt-4">Supported: .dtsx (SSIS), .sql</p>
          </>
        )}
      </div>

      {/* Token Estimation Card */}
      {file && tokenEstimate && (
        <div className="mt-4 p-4 bg-blue-50 border border-blue-200 rounded-lg">
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

      {/* Job Name */}
      <div className="mt-6">
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

      {/* Model Selection */}
      <div className="mt-6">
        <label className="block text-sm font-medium text-gray-700 mb-2">
          AI Model
        </label>
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
      </div>

      {/* Conversion Instructions - Always visible */}
      <div className="mt-6">
        <label className="block text-sm font-medium text-gray-700 mb-2">
          Conversion Instructions (Optional)
        </label>
        <textarea
          value={conversionInstructions}
          onChange={(e) => setConversionInstructions(e.target.value)}
          placeholder="Add specific instructions for the conversion (e.g., 'Use Unity Catalog table catalog.schema.table for the destination', 'Implement error handling with try/except blocks')"
          rows={3}
          className="w-full px-4 py-2 border rounded-lg focus:ring-2 focus:ring-[#ff6600] focus:border-transparent"
        />
        <p className="text-xs text-gray-500 mt-1">
          These instructions will be passed to the AI model to guide the conversion
        </p>
      </div>

      {/* Advanced Options */}
      <div className="mt-6">
        <button
          onClick={() => setShowAdvanced(!showAdvanced)}
          className="flex items-center gap-2 text-sm font-medium text-gray-700 hover:text-gray-900"
        >
          {showAdvanced ? <ChevronUp size={16} /> : <ChevronDown size={16} />}
          Advanced Options
        </button>

        {showAdvanced && (
          <div className="mt-4 space-y-6 p-4 bg-gray-50 rounded-lg">
            {/* Reference Files */}
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Reference Files (Optional)
              </label>
              <p className="text-xs text-gray-500 mb-3">
                Upload documentation, example code, or style guides to help inform the conversion
              </p>

              {/* Uploaded reference files */}
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

              {/* Add reference file button */}
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
      </div>

      {/* Auto-run Option */}
      <div className="mt-6">
        <label className="flex items-center gap-3 cursor-pointer">
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
      </div>

      {/* Submit */}
      <div className="mt-8 flex gap-4">
        <button
          onClick={() => uploadMutation.mutate()}
          disabled={!file || uploadMutation.isPending}
          className="flex-1 px-4 py-3 bg-[#ff6600] text-white rounded-lg hover:bg-[#e55c00] disabled:opacity-50 disabled:cursor-not-allowed"
        >
          {uploadMutation.isPending
            ? (autoRun ? 'Uploading & Starting...' : 'Uploading...')
            : (autoRun ? 'Upload & Convert' : 'Create Conversion Job')}
        </button>
      </div>

      {uploadMutation.isError && (
        <div className="mt-4 p-4 bg-red-50 text-red-700 rounded-lg">
          Error: {(uploadMutation.error as Error).message}
        </div>
      )}
    </div>
  )
}

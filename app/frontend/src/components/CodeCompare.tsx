import { useState } from 'react'
import { Code, FileCode2, Copy, Check, ChevronDown, ChevronRight } from 'lucide-react'

interface OutputFile {
  name: string
  content: string
  type: string
}

interface CodeCompareProps {
  sourceFile: string
  sourceContent: string
  outputFiles: OutputFile[]
}

export default function CodeCompare({ sourceFile, sourceContent, outputFiles }: CodeCompareProps) {
  const [selectedOutput, setSelectedOutput] = useState(0)
  const [copiedSource, setCopiedSource] = useState(false)
  const [copiedOutput, setCopiedOutput] = useState(false)
  const [sourceCollapsed, setSourceCollapsed] = useState(false)
  const [outputCollapsed, setOutputCollapsed] = useState(false)

  const copyToClipboard = async (text: string, type: 'source' | 'output') => {
    await navigator.clipboard.writeText(text)
    if (type === 'source') {
      setCopiedSource(true)
      setTimeout(() => setCopiedSource(false), 2000)
    } else {
      setCopiedOutput(true)
      setTimeout(() => setCopiedOutput(false), 2000)
    }
  }

  const currentOutput = outputFiles[selectedOutput]

  return (
    <div className="bg-white rounded-xl shadow-sm border">
      <div className="border-b p-4">
        <h2 className="text-lg font-semibold flex items-center gap-2">
          <Code size={20} />
          Code Compare
        </h2>
        <p className="text-sm text-gray-500">Side-by-side view of source and converted code</p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 divide-y lg:divide-y-0 lg:divide-x">
        {/* Source Panel */}
        <div className="flex flex-col">
          <div className="flex items-center justify-between p-3 bg-gray-50 border-b">
            <button
              onClick={() => setSourceCollapsed(!sourceCollapsed)}
              className="flex items-center gap-2 font-medium text-gray-700"
            >
              {sourceCollapsed ? <ChevronRight size={16} /> : <ChevronDown size={16} />}
              <FileCode2 size={16} className="text-blue-600" />
              Source: {sourceFile || 'No file'}
            </button>
            <button
              onClick={() => copyToClipboard(sourceContent, 'source')}
              className="p-1.5 hover:bg-gray-200 rounded"
              title="Copy source"
            >
              {copiedSource ? <Check size={16} className="text-green-600" /> : <Copy size={16} />}
            </button>
          </div>
          {!sourceCollapsed && (
            <div className="flex-1 overflow-auto max-h-[500px]">
              <pre className="p-4 text-sm font-mono bg-gray-900 text-gray-100 overflow-x-auto">
                <code>{sourceContent || '// No source content'}</code>
              </pre>
            </div>
          )}
        </div>

        {/* Output Panel */}
        <div className="flex flex-col">
          <div className="flex items-center justify-between p-3 bg-gray-50 border-b">
            <div className="flex items-center gap-2">
              <button
                onClick={() => setOutputCollapsed(!outputCollapsed)}
                className="flex items-center gap-2 font-medium text-gray-700"
              >
                {outputCollapsed ? <ChevronRight size={16} /> : <ChevronDown size={16} />}
                <FileCode2 size={16} className="text-green-600" />
                Output:
              </button>
              {outputFiles.length > 1 && (
                <select
                  value={selectedOutput}
                  onChange={(e) => setSelectedOutput(Number(e.target.value))}
                  className="text-sm border rounded px-2 py-1"
                >
                  {outputFiles.map((file, i) => (
                    <option key={i} value={i}>{file.name}</option>
                  ))}
                </select>
              )}
              {outputFiles.length === 1 && (
                <span className="text-sm text-gray-600">{currentOutput?.name}</span>
              )}
            </div>
            <button
              onClick={() => copyToClipboard(currentOutput?.content || '', 'output')}
              className="p-1.5 hover:bg-gray-200 rounded"
              title="Copy output"
            >
              {copiedOutput ? <Check size={16} className="text-green-600" /> : <Copy size={16} />}
            </button>
          </div>
          {!outputCollapsed && (
            <div className="flex-1 overflow-auto max-h-[500px]">
              <pre className="p-4 text-sm font-mono bg-gray-900 text-gray-100 overflow-x-auto">
                <code>{currentOutput?.content || '// No output yet'}</code>
              </pre>
            </div>
          )}
        </div>
      </div>

      {/* Output file type badges */}
      {outputFiles.length > 0 && (
        <div className="p-3 bg-gray-50 border-t flex gap-2 flex-wrap">
          {outputFiles.map((file, i) => (
            <button
              key={i}
              onClick={() => setSelectedOutput(i)}
              className={`px-3 py-1 text-xs font-medium rounded-full ${
                selectedOutput === i
                  ? 'bg-[#ff6600] text-white'
                  : 'bg-gray-200 text-gray-700 hover:bg-gray-300'
              }`}
            >
              {file.name}
              <span className="ml-1 opacity-70">({file.type})</span>
            </button>
          ))}
        </div>
      )}
    </div>
  )
}

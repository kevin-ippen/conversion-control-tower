import { useState, useCallback } from 'react'
import { Folder, FolderOpen, FileCode2, FileText, ChevronRight, ChevronDown, Loader2, AlertCircle } from 'lucide-react'
import axios from 'axios'

interface BrowseItem {
  name: string
  path: string
  is_directory: boolean
  size: number
  modified_at?: string
}

interface FileBrowserProps {
  source: 'workspace' | 'volume' | 'repo'
  rootPath: string
  onSelect: (path: string, name: string) => void
  allowedExtensions?: string[]
}

interface TreeNode extends BrowseItem {
  children?: TreeNode[]
  loaded: boolean
  expanded: boolean
}

const FILE_ICONS: Record<string, typeof FileCode2> = {
  '.dtsx': FileCode2,
  '.sql': FileCode2,
  '.xml': FileText,
  '.py': FileCode2,
  '.txt': FileText,
}

function getFileIcon(name: string) {
  const ext = name.includes('.') ? `.${name.split('.').pop()?.toLowerCase()}` : ''
  const Icon = FILE_ICONS[ext] || FileText
  return <Icon size={16} className="text-gray-500 flex-shrink-0" />
}

function formatSize(bytes: number): string {
  if (bytes === 0) return ''
  if (bytes < 1024) return `${bytes} B`
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`
}

export default function FileBrowser({ source, rootPath, onSelect }: FileBrowserProps) {
  const [nodes, setNodes] = useState<TreeNode[]>([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [selectedPath, setSelectedPath] = useState<string | null>(null)
  const [initialized, setInitialized] = useState(false)

  const endpoint = source === 'workspace' ? '/api/files/browse-workspace'
    : source === 'volume' ? '/api/files/browse-volume'
    : '/api/files/browse-repo'

  const loadDirectory = useCallback(async (path: string): Promise<TreeNode[]> => {
    const { data } = await axios.post(endpoint, { path })
    return (data.items || []).map((item: BrowseItem) => ({
      ...item,
      children: item.is_directory ? [] : undefined,
      loaded: false,
      expanded: false,
    }))
  }, [endpoint])

  const initialize = useCallback(async () => {
    if (initialized || !rootPath) return
    setLoading(true)
    setError(null)
    try {
      const children = await loadDirectory(rootPath)
      setNodes(children)
      setInitialized(true)
    } catch (e: any) {
      setError(e.response?.data?.detail || e.message || 'Failed to browse')
    } finally {
      setLoading(false)
    }
  }, [rootPath, initialized, loadDirectory])

  // Auto-initialize on mount
  if (!initialized && rootPath && !loading) {
    initialize()
  }

  const toggleExpand = async (nodePath: string) => {
    const updateNodes = (items: TreeNode[]): TreeNode[] => {
      return items.map(node => {
        if (node.path === nodePath && node.is_directory) {
          return { ...node, expanded: !node.expanded }
        }
        if (node.children) {
          return { ...node, children: updateNodes(node.children) }
        }
        return node
      })
    }

    // If not loaded yet, fetch children first
    const findNode = (items: TreeNode[]): TreeNode | null => {
      for (const item of items) {
        if (item.path === nodePath) return item
        if (item.children) {
          const found = findNode(item.children)
          if (found) return found
        }
      }
      return null
    }

    const node = findNode(nodes)
    if (node && !node.loaded && node.is_directory) {
      try {
        const children = await loadDirectory(nodePath)
        const updateWithChildren = (items: TreeNode[]): TreeNode[] => {
          return items.map(n => {
            if (n.path === nodePath) {
              return { ...n, children, loaded: true, expanded: true }
            }
            if (n.children) {
              return { ...n, children: updateWithChildren(n.children) }
            }
            return n
          })
        }
        setNodes(updateWithChildren(nodes))
        return
      } catch (e) {
        // Fall through to just toggle
      }
    }

    setNodes(updateNodes(nodes))
  }

  const handleSelect = (path: string, name: string) => {
    setSelectedPath(path)
    onSelect(path, name)
  }

  const renderNode = (node: TreeNode, depth: number = 0) => {
    const isSelected = selectedPath === node.path

    if (node.is_directory) {
      return (
        <div key={node.path}>
          <button
            onClick={() => toggleExpand(node.path)}
            className="w-full flex items-center gap-2 px-2 py-1.5 hover:bg-gray-100 rounded text-sm text-left"
            style={{ paddingLeft: `${depth * 20 + 8}px` }}
          >
            {node.expanded ? <ChevronDown size={14} /> : <ChevronRight size={14} />}
            {node.expanded ? <FolderOpen size={16} className="text-yellow-500" /> : <Folder size={16} className="text-yellow-500" />}
            <span className="truncate">{node.name}</span>
          </button>
          {node.expanded && node.children?.map(child => renderNode(child, depth + 1))}
        </div>
      )
    }

    return (
      <button
        key={node.path}
        onClick={() => handleSelect(node.path, node.name)}
        className={`w-full flex items-center gap-2 px-2 py-1.5 rounded text-sm text-left ${
          isSelected ? 'bg-orange-50 border border-[#ff6600]' : 'hover:bg-gray-100'
        }`}
        style={{ paddingLeft: `${depth * 20 + 28}px` }}
      >
        {getFileIcon(node.name)}
        <span className="truncate flex-1">{node.name}</span>
        {node.size > 0 && (
          <span className="text-xs text-gray-400 flex-shrink-0">{formatSize(node.size)}</span>
        )}
      </button>
    )
  }

  if (loading && !initialized) {
    return (
      <div className="flex items-center justify-center py-8 text-gray-500">
        <Loader2 size={20} className="animate-spin mr-2" />
        Loading...
      </div>
    )
  }

  if (error) {
    return (
      <div className="flex items-center gap-2 py-4 px-3 text-red-600 text-sm">
        <AlertCircle size={16} />
        {error}
      </div>
    )
  }

  if (nodes.length === 0 && initialized) {
    return (
      <div className="py-4 px-3 text-gray-500 text-sm text-center">
        No files found at this path
      </div>
    )
  }

  return (
    <div className="border rounded-lg max-h-[300px] overflow-auto bg-white">
      <div className="p-1">
        {nodes.map(node => renderNode(node))}
      </div>
    </div>
  )
}

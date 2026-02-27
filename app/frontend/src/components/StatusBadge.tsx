import { CheckCircle2, XCircle, Clock, Loader2 } from 'lucide-react'
import clsx from 'clsx'

interface StatusBadgeProps {
  status: string
  size?: 'sm' | 'md' | 'lg'
}

const statusConfig: Record<string, { color: string; bgColor: string; icon: React.ReactNode }> = {
  pending: {
    color: 'text-gray-600',
    bgColor: 'bg-gray-100',
    icon: <Clock size={14} />,
  },
  parsing: {
    color: 'text-blue-600',
    bgColor: 'bg-blue-100',
    icon: <Loader2 size={14} className="animate-spin" />,
  },
  converting: {
    color: 'text-blue-600',
    bgColor: 'bg-blue-100',
    icon: <Loader2 size={14} className="animate-spin" />,
  },
  validating: {
    color: 'text-purple-600',
    bgColor: 'bg-purple-100',
    icon: <Loader2 size={14} className="animate-spin" />,
  },
  completed: {
    color: 'text-green-600',
    bgColor: 'bg-green-100',
    icon: <CheckCircle2 size={14} />,
  },
  failed: {
    color: 'text-red-600',
    bgColor: 'bg-red-100',
    icon: <XCircle size={14} />,
  },
  cancelled: {
    color: 'text-gray-600',
    bgColor: 'bg-gray-100',
    icon: <XCircle size={14} />,
  },
}

export default function StatusBadge({ status, size = 'md' }: StatusBadgeProps) {
  const config = statusConfig[status] || statusConfig.pending

  const sizeClasses = {
    sm: 'px-2 py-0.5 text-xs',
    md: 'px-2.5 py-1 text-sm',
    lg: 'px-3 py-1.5 text-base',
  }

  return (
    <span
      className={clsx(
        'inline-flex items-center gap-1.5 font-medium rounded-full',
        config.color,
        config.bgColor,
        sizeClasses[size]
      )}
    >
      {config.icon}
      <span className="capitalize">{status}</span>
    </span>
  )
}

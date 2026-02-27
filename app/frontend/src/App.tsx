import { Routes, Route, NavLink } from 'react-router-dom'
import {
  LayoutDashboard,
  FileCode2,
  Rocket,
  PlusCircle,
} from 'lucide-react'
import Dashboard from './pages/Dashboard'
import Conversions from './pages/Conversions'
import ConversionDetail from './pages/ConversionDetail'
import NewConversion from './pages/NewConversion'
import Promote from './pages/Promote'

function App() {
  return (
    <div className="min-h-screen flex">
      {/* Sidebar */}
      <aside className="w-64 bg-[#1b3a57] text-white flex flex-col">
        <div className="p-4 border-b border-white/10">
          <h1 className="text-xl font-bold">Conversion Control Tower</h1>
          <p className="text-sm text-white/70">SQL → Databricks</p>
        </div>

        <nav className="flex-1 p-4 space-y-1">
          <NavItem to="/" icon={<LayoutDashboard size={20} />} label="Dashboard" />
          <NavItem to="/conversions" icon={<FileCode2 size={20} />} label="Conversions" />
          <NavItem to="/new" icon={<PlusCircle size={20} />} label="New Conversion" />
          <NavItem to="/promote" icon={<Rocket size={20} />} label="Promote" />
        </nav>

        <div className="p-4 border-t border-white/10 text-xs text-white/50">
          Powered by Databricks FMAPI
        </div>
      </aside>

      {/* Main Content */}
      <main className="flex-1 overflow-auto">
        <Routes>
          <Route path="/" element={<Dashboard />} />
          <Route path="/conversions" element={<Conversions />} />
          <Route path="/conversions/:jobId" element={<ConversionDetail />} />
          <Route path="/new" element={<NewConversion />} />
          <Route path="/promote" element={<Promote />} />
        </Routes>
      </main>
    </div>
  )
}

function NavItem({ to, icon, label }: { to: string; icon: React.ReactNode; label: string }) {
  return (
    <NavLink
      to={to}
      className={({ isActive }) =>
        `flex items-center gap-3 px-3 py-2 rounded-lg transition-colors ${
          isActive
            ? 'bg-white/10 text-white'
            : 'text-white/70 hover:bg-white/5 hover:text-white'
        }`
      }
    >
      {icon}
      <span>{label}</span>
    </NavLink>
  )
}

export default App

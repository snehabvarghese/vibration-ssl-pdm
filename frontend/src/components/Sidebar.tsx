import { NavLink } from 'react-router-dom';
import { useTheme } from '../context/ThemeContext';

const links = [
  { to: '/analyse', label: 'Vibration', icon: '📊' },
  { to: '/motor', label: 'Motor & Temp', icon: '⚡' },
  { to: '/history', label: 'History', icon: '📋' },
  { to: '/live', label: 'Live', icon: '📡' },
];

export default function Sidebar() {
  const { theme, toggle } = useTheme();

  return (
    <aside className="fixed top-0 left-0 h-screen w-56 bg-surface-1 border-r border-white/5 flex flex-col z-40">
      {/* Logo / Brand */}
      <div className="px-5 py-6 border-b border-white/5">
        <div className="flex items-center gap-2.5">
          <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-brand-400 to-brand-600 flex items-center justify-center text-white font-bold text-sm shadow-md">
            V
          </div>
          <div>
            <h1 className="text-sm font-bold text-white tracking-tight leading-none">
              VibrationPdM
            </h1>
            <p className="text-[10px] text-gray-500 mt-0.5 font-medium">SSL Diagnostics</p>
          </div>
        </div>
      </div>

      {/* Nav links */}
      <nav className="flex-1 px-3 py-4 space-y-1">
        {links.map((link) => (
          <NavLink
            key={link.to}
            to={link.to}
            className={({ isActive }) =>
              `flex items-center gap-3 px-3 py-2.5 rounded-lg text-sm font-medium transition-all duration-200 ${
                isActive
                  ? 'bg-brand-500/15 text-brand-300 glow-ring'
                  : 'text-gray-400 hover:text-gray-200 hover:bg-white/5'
              }`
            }
          >
            <span className="text-base">{link.icon}</span>
            {link.label}
          </NavLink>
        ))}
      </nav>

      {/* Footer: theme toggle */}
      <div className="px-3 py-4 border-t border-white/5">
        <button
          onClick={toggle}
          className="flex items-center gap-2 w-full px-3 py-2 rounded-lg text-sm text-gray-400 hover:text-gray-200 hover:bg-white/5 transition-colors"
        >
          <span className="text-base">{theme === 'dark' ? '☀️' : '🌙'}</span>
          {theme === 'dark' ? 'Light mode' : 'Dark mode'}
        </button>
      </div>
    </aside>
  );
}

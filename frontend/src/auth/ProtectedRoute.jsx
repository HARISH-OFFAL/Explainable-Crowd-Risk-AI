import { Navigate, Outlet, useLocation } from 'react-router-dom'
import { useAuth } from './AuthContext'
import { ROLES } from '../config/demoAuth'

export function ProtectedRoute({ role }) {
  const { isAuthenticated, role: currentRole } = useAuth()
  const location = useLocation()
  if (!isAuthenticated) return <Navigate to="/login" replace state={{ from: location }} />
  if (role && currentRole !== role) return <Navigate to={currentRole === ROLES.AUTHORITY ? '/authority/dashboard' : '/organizer/dashboard'} replace />
  return <Outlet />
}

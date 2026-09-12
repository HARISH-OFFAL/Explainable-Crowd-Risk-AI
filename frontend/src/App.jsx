import {
  BrowserRouter,
  Routes,
  Route,
} from 'react-router-dom'

import Dashboard from './pages/Dashboard'
import EventRegistration from './pages/EventRegistration'
import RiskResult from './pages/RiskResult'
import Events from './pages/Events'
import Documents from './pages/Documents'
import ApplicationStatus from './pages/ApplicationStatus'
import CustomCursor from './components/CustomCursor'
import Phase1Summary from './pages/Phase1Summary'
import MonitoringLaunch from './pages/MonitoringLaunch'
import FlowIntelligence from './pages/FlowIntelligence'
import DigitalTwin from './pages/DigitalTwin'
import CrowdTimeMachine from './pages/CrowdTimeMachine'
import ResponseCommander from './pages/ResponseCommander'
import Login from './pages/Login'
import { AuthProvider } from './auth/AuthContext'
import { ProtectedRoute } from './auth/ProtectedRoute'
import { ROLES } from './config/demoAuth'

import './App.css'

function App() {
  return (
    <AuthProvider>
      <BrowserRouter>
        <CustomCursor />
        <Routes>
          <Route path="/" element={<Login />} />
          <Route path="/login" element={<Login />} />

          <Route element={<ProtectedRoute role={ROLES.ORGANIZER} />}>
            <Route path="/dashboard" element={<Dashboard />} />
            <Route path="/organizer/dashboard" element={<Dashboard />} />
            <Route path="/events" element={<Events />} />
            <Route path="/events/register" element={<EventRegistration />} />
            <Route path="/events/:eventId/risk" element={<RiskResult />} />
            <Route path="/documents" element={<Documents />} />
            <Route path="/organizer/documents" element={<Documents />} />
            <Route path="/application-status" element={<ApplicationStatus />} />
            <Route path="/organizer/application-status" element={<ApplicationStatus />} />
            <Route path="/organizer/events/:eventId/phase1-summary" element={<Phase1Summary />} />
            <Route path="/organizer/events/:eventId/monitoring" element={<MonitoringLaunch />} />
            <Route path="/organizer/events/:eventId/flow-intelligence/:flowAnalysisId" element={<FlowIntelligence />} />
            <Route path="/organizer/events/:eventId/digital-twin/:sessionId" element={<DigitalTwin />} />
            <Route path="/organizer/events/:eventId/crowd-time-machine" element={<CrowdTimeMachine />} />
            <Route path="/organizer/events/:eventId/response-commander" element={<ResponseCommander />} />
          </Route>

          <Route element={<ProtectedRoute role={ROLES.AUTHORITY} />}>
            <Route path="/authority/dashboard" element={<Dashboard />} />
            <Route path="/authority/events" element={<Events />} />
            <Route path="/authority/events/:eventId/risk" element={<RiskResult />} />
            <Route path="/authority/documents" element={<Documents />} />
          </Route>
        </Routes>
      </BrowserRouter>
    </AuthProvider>
  )
}

export default App

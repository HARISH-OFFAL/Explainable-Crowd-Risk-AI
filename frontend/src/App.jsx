import {
  BrowserRouter,
  Routes,
  Route,
} from 'react-router-dom'

import Home from './pages/Home'
import Dashboard from './pages/Dashboard'
import EventRegistration from './pages/EventRegistration'
import RiskResult from './pages/RiskResult'
import Events from './pages/Events'
import Documents from './pages/Documents'
import ApplicationStatus from './pages/ApplicationStatus'
import CustomCursor from './components/CustomCursor'
import Phase1Summary from './pages/Phase1Summary'

import './App.css'

function App() {
  return (
    <BrowserRouter>
      <CustomCursor />
      <Routes>
        <Route
          path="/"
          element={<Home />}
        />

        <Route
          path="/dashboard"
          element={<Dashboard />}
        />

        <Route path="/organizer/dashboard" element={<Dashboard />} />
        <Route path="/authority/dashboard" element={<Dashboard />} />

        <Route
          path="/events"
          element={<Events />}
        />

        <Route path="/authority/events" element={<Events />} />

        <Route
          path="/events/register"
          element={<EventRegistration />}
        />

        <Route
          path="/events/:eventId/risk"
          element={<RiskResult />}
        />

        <Route path="/authority/events/:eventId/risk" element={<RiskResult />} />

        <Route
          path="/documents"
          element={<Documents />}
        />

        <Route path="/organizer/documents" element={<Documents />} />
        <Route path="/authority/documents" element={<Documents />} />

        <Route
          path="/application-status"
          element={<ApplicationStatus />}
        />

        <Route path="/organizer/application-status" element={<ApplicationStatus />} />

        <Route path="/organizer/events/:eventId/phase1-summary" element={<Phase1Summary />} />
      </Routes>
    </BrowserRouter>
  )
}

export default App

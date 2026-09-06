import { useState } from 'react'
import { useNavigate } from 'react-router-dom'

import Navbar from '../components/Navbar'

const initialForm = {
  event_name: '',
  location: '',
  event_datetime: '',
  expected_crowd_size: '',
  venue_capacity: '',
  entry_gates: '',
  exit_gates: '',
  emergency_exits: '',
  event_duration_minutes: '',
}

function EventRegistration() {
  const navigate = useNavigate()

  const [formData, setFormData] = useState(initialForm)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')

  const handleChange = (event) => {
    const { name, value } = event.target

    setFormData((current) => ({
      ...current,
      [name]: value,
    }))

    setError('')
  }

  const handleSubmit = async (event) => {
    event.preventDefault()

    setLoading(true)
    setError('')

    const payload = {
      event_name: formData.event_name.trim(),
      location: formData.location.trim(),
      event_datetime: formData.event_datetime,

      expected_crowd_size: Number(
        formData.expected_crowd_size
      ),

      venue_capacity: Number(
        formData.venue_capacity
      ),

      entry_gates: Number(
        formData.entry_gates
      ),

      exit_gates: Number(
        formData.exit_gates
      ),

      emergency_exits: Number(
        formData.emergency_exits
      ),

      event_duration_minutes: Number(
        formData.event_duration_minutes
      ),
    }

    try {
      const response = await fetch(
        'http://127.0.0.1:8000/events',
        {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify(payload),
        }
      )

      const data = await response.json()

      if (!response.ok) {
        let message = 'Unable to register event.'

        if (typeof data.detail === 'string') {
          message = data.detail
        }

        if (Array.isArray(data.detail)) {
          message = data.detail
            .map((item) => item.msg)
            .join(', ')
        }

        throw new Error(message)
      }

      navigate(`/events/${data.event_id}/risk`, {
        state: {
          result: data,
        },
      })
    } catch (requestError) {
      console.error(requestError)

      if (requestError instanceof TypeError) {
        setError(
          'Unable to connect to the backend server. Make sure FastAPI is running.'
        )
      } else {
        setError(
          requestError.message ||
            'Something went wrong while registering the event.'
        )
      }
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="app registration-page">
      <div className="background-grid" />
      <div className="glow glow-one" />
      <div className="glow glow-two" />

      <Navbar />

      <main className="page-main">
        <section className="page-heading">
          <div className="eyebrow">
            NEW EVENT
          </div>

          <h1>
            Register <span>Event</span>
          </h1>

          <p>
            Enter event and crowd details to generate the initial
            pre-event risk assessment.
          </p>
        </section>

        <section className="form-layout">
          <form
            className="event-form glass-card"
            onSubmit={handleSubmit}
          >
            <div className="form-section-title">
              <h3>Event Details</h3>

              <p>
                Basic information about the crowd gathering.
              </p>
            </div>

            <div className="form-grid">
              <div className="form-group">
                <label htmlFor="event_name">
                  Event Name
                </label>

                <input
                  id="event_name"
                  name="event_name"
                  type="text"
                  placeholder="College Cultural Festival"
                  value={formData.event_name}
                  onChange={handleChange}
                  required
                />
              </div>

              <div className="form-group">
                <label htmlFor="location">
                  Location
                </label>

                <input
                  id="location"
                  name="location"
                  type="text"
                  placeholder="Maharaja Auditorium"
                  value={formData.location}
                  onChange={handleChange}
                  required
                />
              </div>

              <div className="form-group full">
                <label htmlFor="event_datetime">
                  Event Date & Time
                </label>

                <input
                  id="event_datetime"
                  name="event_datetime"
                  type="datetime-local"
                  value={formData.event_datetime}
                  onChange={handleChange}
                  required
                />
              </div>
            </div>

            <div className="form-section-title">
              <h3>Crowd & Venue</h3>

              <p>
                Expected attendance and venue capacity details.
              </p>
            </div>

            <div className="form-grid">
              <div className="form-group">
                <label htmlFor="expected_crowd_size">
                  Expected Crowd Size
                </label>

                <input
                  id="expected_crowd_size"
                  name="expected_crowd_size"
                  type="number"
                  min="1"
                  placeholder="5000"
                  value={formData.expected_crowd_size}
                  onChange={handleChange}
                  required
                />
              </div>

              <div className="form-group">
                <label htmlFor="venue_capacity">
                  Venue Capacity
                </label>

                <input
                  id="venue_capacity"
                  name="venue_capacity"
                  type="number"
                  min="1"
                  placeholder="7000"
                  value={formData.venue_capacity}
                  onChange={handleChange}
                  required
                />
              </div>
            </div>

            <div className="form-section-title">
              <h3>Access & Emergency</h3>

              <p>
                Entry, exit and emergency evacuation arrangements.
              </p>
            </div>

            <div className="form-grid">
              <div className="form-group">
                <label htmlFor="entry_gates">
                  Entry Gates
                </label>

                <input
                  id="entry_gates"
                  name="entry_gates"
                  type="number"
                  min="0"
                  placeholder="4"
                  value={formData.entry_gates}
                  onChange={handleChange}
                  required
                />
              </div>

              <div className="form-group">
                <label htmlFor="exit_gates">
                  Exit Gates
                </label>

                <input
                  id="exit_gates"
                  name="exit_gates"
                  type="number"
                  min="0"
                  placeholder="4"
                  value={formData.exit_gates}
                  onChange={handleChange}
                  required
                />
              </div>

              <div className="form-group">
                <label htmlFor="emergency_exits">
                  Emergency Exits
                </label>

                <input
                  id="emergency_exits"
                  name="emergency_exits"
                  type="number"
                  min="0"
                  placeholder="2"
                  value={formData.emergency_exits}
                  onChange={handleChange}
                  required
                />
              </div>

              <div className="form-group">
                <label htmlFor="event_duration_minutes">
                  Event Duration
                </label>

                <div className="unit-input">
                  <input
                    id="event_duration_minutes"
                    name="event_duration_minutes"
                    type="number"
                    min="1"
                    placeholder="240"
                    value={formData.event_duration_minutes}
                    onChange={handleChange}
                    required
                  />

                  <span>MINUTES</span>
                </div>
              </div>
            </div>

            {error && (
              <div className="form-error">
                <strong>Registration Failed</strong>
                <p>{error}</p>
              </div>
            )}

            <div className="form-actions">
              <button
                className="cancel-btn"
                type="button"
                onClick={() => navigate('/dashboard')}
                disabled={loading}
              >
                Back to Dashboard
              </button>

              <button
                className="primary-btn"
                type="submit"
                disabled={loading}
              >
                {loading
                  ? 'Assessing Event...'
                  : 'Register & Assess Risk →'}
              </button>
            </div>
          </form>

          <aside className="guide-card glass-card">
            <h3>What We Assess</h3>

            <p>
              The risk engine uses these planning details to explain
              possible crowd-safety concerns.
            </p>

            <ul className="guide-list">
              <li>Crowd vs venue capacity</li>
              <li>Entry and exit availability</li>
              <li>Emergency exit readiness</li>
              <li>Event duration</li>
            </ul>
          </aside>
        </section>
      </main>

      <footer>
        CrowdGuard · Event Registration
      </footer>
    </div>
  )
}

export default EventRegistration
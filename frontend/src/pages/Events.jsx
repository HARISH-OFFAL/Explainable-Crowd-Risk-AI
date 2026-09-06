import { useEffect, useState } from 'react'
import { useLocation, useNavigate } from 'react-router-dom'
import Navbar from '../components/Navbar'

const API_URL = 'http://127.0.0.1:8000'

const emptyEditForm = {
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

function toDateTimeInput(value) {
  return value ? String(value).slice(0, 16) : ''
}

function Events() {
  const navigate = useNavigate()
  const authority = useLocation().pathname.startsWith('/authority')

  const [events, setEvents] = useState([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')
  const [selected, setSelected] = useState(null)
  const [editing, setEditing] = useState(null)
  const [editForm, setEditForm] = useState(emptyEditForm)
  const [savingEdit, setSavingEdit] = useState(false)
  const [toast, setToast] = useState('')

  const loadEvents = async () => {
    setLoading(true)
    setError('')

    try {
      const response = await fetch(`${API_URL}/events`)
      const data = await response.json()

      if (!response.ok) {
        throw new Error(data.detail || 'Unable to load events.')
      }

      setEvents(Array.isArray(data) ? data : [])
    } catch (requestError) {
      setError(
        requestError instanceof TypeError
          ? 'Unable to connect to the backend server.'
          : requestError.message
      )
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    loadEvents()
  }, [])

  const viewEvent = async (id) => {
    try {
      const response = await fetch(`${API_URL}/events/${id}`)
      const data = await response.json()

      if (!response.ok) {
        throw new Error(data.detail || 'Unable to load event.')
      }

      navigate(`${authority ? '/authority' : ''}/events/${id}/risk`, {
        state: {
          result: {
            event_id: data.id,
            event_name: data.event_name,
            pre_event_risk: data.pre_event_risk,
          },
        },
      })
    } catch (requestError) {
      setError(requestError.message)
    }
  }

  const startEdit = (event) => {
    setEditing(event)
    setEditForm({
      event_name: event.event_name || '',
      location: event.location || '',
      event_datetime: toDateTimeInput(event.event_datetime),
      expected_crowd_size: event.expected_crowd_size || '',
      venue_capacity: event.venue_capacity || '',
      entry_gates: event.entry_gates || '',
      exit_gates: event.exit_gates || '',
      emergency_exits: event.emergency_exits || '',
      event_duration_minutes: event.event_duration_minutes || '',
    })
    setError('')
  }

  const updateEditForm = (event) => {
    const { name, value } = event.target
    setEditForm((current) => ({ ...current, [name]: value }))
  }

  const saveEdit = async (event) => {
    event.preventDefault()
    if (!editing) return

    setSavingEdit(true)
    setError('')

    const payload = {
      event_name: editForm.event_name.trim(),
      location: editForm.location.trim(),
      event_datetime: editForm.event_datetime,
      expected_crowd_size: Number(editForm.expected_crowd_size),
      venue_capacity: Number(editForm.venue_capacity),
      entry_gates: Number(editForm.entry_gates),
      exit_gates: Number(editForm.exit_gates),
      emergency_exits: Number(editForm.emergency_exits),
      event_duration_minutes: Number(editForm.event_duration_minutes),
    }

    try {
      const response = await fetch(`${API_URL}/events/${editing.id}`, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      })
      const data = await response.json()

      if (!response.ok) {
        const detail = Array.isArray(data.detail)
          ? data.detail.map((item) => item.msg).join(', ')
          : data.detail
        throw new Error(detail || 'Unable to update event.')
      }

      setEvents((current) =>
        current.map((item) =>
          item.id === editing.id ? { ...item, ...payload } : item
        )
      )
      setEditing(null)
      setToast('Event updated successfully.')
      setTimeout(() => setToast(''), 3000)
    } catch (requestError) {
      setError(requestError.message)
    } finally {
      setSavingEdit(false)
    }
  }

  const deleteEvent = async () => {
    if (!selected) return

    try {
      const response = await fetch(`${API_URL}/events/${selected.id}`, {
        method: 'DELETE',
      })
      const data = await response.json()

      if (!response.ok) {
        throw new Error(
          typeof data.detail === 'string'
            ? data.detail
            : 'Unable to delete event.'
        )
      }

      setEvents((current) => current.filter((item) => item.id !== selected.id))
      setSelected(null)
      setToast('Event deleted successfully.')
      setTimeout(() => setToast(''), 3000)
    } catch (requestError) {
      setError(requestError.message)
    }
  }

  return (
    <div className="app">
      <div className="background-grid" />
      <div className="glow glow-one" />
      <Navbar />

      <main className="page-main">
        <div className="events-toolbar">
          <div>
            <div className="eyebrow">
              {authority ? 'AUTHORITY REVIEW QUEUE' : 'EVENT MANAGEMENT'}
            </div>
            <h1>{authority ? 'Submitted Events' : 'Registered Events'}</h1>
            <p>
              {authority
                ? 'Select an event to inspect its details and existing risk result.'
                : 'View and manage the event submissions you registered.'}
            </p>
          </div>

          {!authority && (
            <button className="primary-btn" onClick={() => navigate('/events/register')}>
              ＋ Register Event
            </button>
          )}
        </div>

        {error && <div className="form-error">{error}</div>}

        {loading ? (
          <div className="empty-card glass-card">Loading events…</div>
        ) : events.length === 0 ? (
          <div className="empty-card glass-card">
            <h3>No events registered yet</h3>
            <p>{authority ? 'Submitted events will appear here.' : 'Create your first event to begin.'}</p>
            {!authority && (
              <button className="primary-btn" onClick={() => navigate('/events/register')}>
                Register Event
              </button>
            )}
          </div>
        ) : (
          <div className="event-list">
            {events.map((item) => (
              <article className="event-row glass-card" key={item.id}>
                <div>
                  <h3>{item.event_name}</h3>
                  <p>{item.location}</p>
                </div>

                <div className="event-meta">
                  <span>{new Date(item.event_datetime).toLocaleString()}</span>
                  <span>
                    {item.expected_crowd_size.toLocaleString()} expected ·{' '}
                    {item.venue_capacity.toLocaleString()} capacity
                  </span>
                </div>

                <div className="event-meta">
                  <span>Event ID</span>
                  <strong>#{item.id}</strong>
                </div>

                <div className="event-actions">
                  <button className="small-btn" onClick={() => viewEvent(item.id)}>
                    View
                  </button>

                  {!authority && (
                    <>
                      <button className="small-btn" onClick={() => navigate(`/organizer/events/${item.id}/phase1-summary`)}>
                        Phase 1
                      </button>
                      <button className="small-btn" onClick={() => startEdit(item)}>
                        Edit
                      </button>
                      <button className="small-btn delete" onClick={() => setSelected(item)}>
                        Delete
                      </button>
                    </>
                  )}
                </div>
              </article>
            ))}
          </div>
        )}
      </main>

      <footer>CrowdGuard · {authority ? 'Authority event review' : 'Event management'}</footer>

      {selected && !authority && (
        <div className="modal-backdrop" onClick={() => setSelected(null)}>
          <div className="modal glass-card" onClick={(event) => event.stopPropagation()}>
            <h2>Delete Event?</h2>
            <p>Are you sure you want to delete this event? This action cannot be undone.</p>
            <div className="modal-event">
              <strong>{selected.event_name}</strong>
              <span>Event ID #{selected.id}</span>
            </div>
            <div className="modal-actions">
              <button className="secondary-btn" onClick={() => setSelected(null)}>Cancel</button>
              <button className="primary-btn danger-btn" onClick={deleteEvent}>Delete Event</button>
            </div>
          </div>
        </div>
      )}

      {editing && !authority && (
        <div className="modal-backdrop" onClick={() => setEditing(null)}>
          <form className="modal glass-card event-edit-modal" onSubmit={saveEdit} onClick={(event) => event.stopPropagation()}>
            <h2>Edit Event</h2>
            <p>Update your event details. The existing system will recalculate its risk result.</p>

            <div className="edit-form-grid">
              {[
                ['event_name', 'Event Name', 'text'],
                ['location', 'Location', 'text'],
                ['event_datetime', 'Event Date & Time', 'datetime-local'],
                ['expected_crowd_size', 'Expected Crowd Size', 'number'],
                ['venue_capacity', 'Venue Capacity', 'number'],
                ['entry_gates', 'Entry Gates', 'number'],
                ['exit_gates', 'Exit Gates', 'number'],
                ['emergency_exits', 'Emergency Exits', 'number'],
                ['event_duration_minutes', 'Event Duration (minutes)', 'number'],
              ].map(([name, label, type]) => (
                <label key={name}>
                  <span>{label}</span>
                  <input
                    name={name}
                    type={type}
                    min={type === 'number' ? (name.includes('gates') || name === 'emergency_exits' ? 0 : 1) : undefined}
                    value={editForm[name]}
                    onChange={updateEditForm}
                    required
                  />
                </label>
              ))}
            </div>

            <div className="modal-actions">
              <button type="button" className="secondary-btn" onClick={() => setEditing(null)}>Cancel</button>
              <button type="submit" className="primary-btn" disabled={savingEdit}>
                {savingEdit ? 'Saving…' : 'Save Changes'}
              </button>
            </div>
          </form>
        </div>
      )}

      {toast && <div className="toast">{toast}</div>}
    </div>
  )
}

export default Events

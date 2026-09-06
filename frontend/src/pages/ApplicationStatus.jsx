import { useEffect, useState } from 'react'
import Navbar from '../components/Navbar'

const API_URL = 'http://127.0.0.1:8000'

function ApplicationStatus() {
  const [events, setEvents] = useState([])
  const [selectedEventId, setSelectedEventId] = useState('')
  const [eventDetails, setEventDetails] = useState(null)
  const [documents, setDocuments] = useState([])

  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')

  useEffect(() => {
    fetchEvents()
  }, [])

  const fetchEvents = async () => {
    try {
      const response = await fetch(`${API_URL}/events`)
      const data = await response.json()

      if (!response.ok) {
        throw new Error('Unable to load events.')
      }

      setEvents(Array.isArray(data) ? data : [])
    } catch (requestError) {
      setError(
        requestError.message ||
          'Unable to connect to backend.'
      )
    }
  }

  const checkStatus = async () => {
    if (!selectedEventId) {
      setError('Please select your event.')
      return
    }

    try {
      setLoading(true)
      setError('')

      const [eventResponse, documentResponse] =
        await Promise.all([
          fetch(`${API_URL}/events/${selectedEventId}`),
          fetch(
            `${API_URL}/events/${selectedEventId}/documents`
          ),
        ])

      const eventData = await eventResponse.json()
      const documentData = await documentResponse.json()

      if (!eventResponse.ok) {
        throw new Error(
          eventData.detail || 'Unable to load event.'
        )
      }

      if (!documentResponse.ok) {
        throw new Error(
          documentData.detail ||
            'Unable to load document status.'
        )
      }

      setEventDetails(eventData)

      setDocuments(
        Array.isArray(documentData.documents)
          ? documentData.documents
          : []
      )
    } catch (requestError) {
      setError(
        requestError.message ||
          'Unable to check application status.'
      )
    } finally {
      setLoading(false)
    }
  }

  const getStatusClass = (status) => {
    switch (status) {
      case 'Approved':
        return 'status-approved'

      case 'Rejected':
        return 'status-rejected'

      case 'Under Review':
        return 'status-review'

      case 'Additional Information Required':
        return 'status-info'

      default:
        return 'status-pending'
    }
  }

  return (
    <div className="app">
      <div className="background-grid" />
      <div className="glow glow-one" />

      <Navbar />

      <main className="page-main">
        <section className="page-heading">
          <div className="eyebrow">
            APPLICATION TRACKING
          </div>

          <h1>
            Track <span>Application Status</span>
          </h1>

          <p>
            Check your event documents, review status and
            remarks provided by the reviewing officer.
          </p>
        </section>

        <section className="glass-card status-search-card">
          <h2>Find Your Event</h2>

          <p>
            Select your registered event to view its current
            document review status.
          </p>

          <div className="status-search-row">
            <select
              value={selectedEventId}
              onChange={(event) => {
                setSelectedEventId(event.target.value)
                setEventDetails(null)
                setDocuments([])
                setError('')
              }}
            >
              <option value="">
                Select your event
              </option>

              {events.map((event) => (
                <option
                  key={event.id}
                  value={event.id}
                >
                  #{event.id} — {event.event_name}
                </option>
              ))}
            </select>

            <button
              className="primary-btn"
              type="button"
              onClick={checkStatus}
              disabled={loading}
            >
              {loading
                ? 'Checking...'
                : 'Check Status'}
            </button>
          </div>

          {error && (
            <div className="form-error">
              {error}
            </div>
          )}
        </section>

        {eventDetails && (
          <section className="glass-card application-event-card">
            <div>
              <span>EVENT</span>
              <strong>
                {eventDetails.event_name}
              </strong>
            </div>

            <div>
              <span>EVENT ID</span>
              <strong>
                #{eventDetails.id}
              </strong>
            </div>

            <div>
              <span>LOCATION</span>
              <strong>
                {eventDetails.location}
              </strong>
            </div>
          </section>
        )}

        {eventDetails && (
          <section className="application-documents">
            <div className="application-status-heading">
              <h2>Document Review Status</h2>

              <p>
                Current review status of the documents submitted
                for this event.
              </p>
            </div>

            {documents.length === 0 ? (
              <div className="glass-card documents-empty">
                <h3>No Documents Found</h3>

                <p>
                  No documents have been submitted for this
                  event yet.
                </p>
              </div>
            ) : (
              <div className="application-status-grid">
                {documents.map((document) => (
                  <article
                    className="glass-card application-status-card"
                    key={document.id}
                  >
                    <div className="application-status-top">
                      <span className="document-type">
                        {document.document_type}
                      </span>

                      <span
                        className={`application-status-badge ${getStatusClass(
                          document.status
                        )}`}
                      >
                        {document.status || 'Pending'}
                      </span>
                    </div>

                    <h3>
                      {document.document_name}
                    </h3>

                    <p className="application-document-id">
                      Document ID #{document.id}
                    </p>

                    <div className="application-remarks">
                      <span>OFFICER REMARKS</span>

                      <p>
                        {document.remarks ||
                          'No remarks provided yet.'}
                      </p>
                    </div>
                  </article>
                ))}
              </div>
            )}
          </section>
        )}
      </main>

      <footer>
        CrowdGuard · Application Status Tracking
      </footer>
    </div>
  )
}

export default ApplicationStatus
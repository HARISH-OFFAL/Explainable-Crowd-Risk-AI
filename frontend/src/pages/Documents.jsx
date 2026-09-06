import { useEffect, useState } from 'react'
import { useLocation } from 'react-router-dom'

import Navbar from '../components/Navbar'

const API_URL = 'http://127.0.0.1:8000'

const DOCUMENT_TYPES = [
  'Event Application',
  'Permission Document',
  'Venue Details',
  'Crowd Management Plan',
  'Emergency / Evacuation Plan',
]

const DOCUMENT_STATUSES = [
  'Pending',
  'Under Review',
  'Approved',
  'Rejected',
  'Additional Information Required',
]

function Documents() {
  const location = useLocation()

  const authority =
    location.pathname.startsWith('/authority')

  const [events, setEvents] = useState([])
  const [selectedEventId, setSelectedEventId] =
    useState('')
  const [documents, setDocuments] = useState([])

  const [documentType, setDocumentType] =
    useState('')
  const [selectedFile, setSelectedFile] =
    useState(null)

  const [eventsLoading, setEventsLoading] =
    useState(true)
  const [documentsLoading, setDocumentsLoading] =
    useState(false)
  const [uploading, setUploading] =
    useState(false)

  const [error, setError] = useState('')
  const [success, setSuccess] = useState('')

  const [
    reviewingDocumentId,
    setReviewingDocumentId,
  ] = useState(null)

  const [reviewStatus, setReviewStatus] =
    useState('')
  const [reviewRemarks, setReviewRemarks] =
    useState('')

  useEffect(() => {
    fetchEvents()
  }, [])

  useEffect(() => {
    if (selectedEventId) {
      fetchDocuments(selectedEventId)
    } else {
      setDocuments([])
    }
  }, [selectedEventId])

  const fetchEvents = async () => {
    try {
      setEventsLoading(true)
      setError('')

      const response = await fetch(
        `${API_URL}/events`
      )

      const data = await response.json()

      if (!response.ok) {
        throw new Error(
          typeof data.detail === 'string'
            ? data.detail
            : 'Unable to load events.'
        )
      }

      setEvents(
        Array.isArray(data) ? data : []
      )
    } catch (requestError) {
      console.error(requestError)

      setError(
        requestError.message ||
          'Unable to connect to the backend server.'
      )
    } finally {
      setEventsLoading(false)
    }
  }

  const fetchDocuments = async (eventId) => {
    try {
      setDocumentsLoading(true)
      setError('')

      const response = await fetch(
        `${API_URL}/events/${eventId}/documents`
      )

      const data = await response.json()

      if (!response.ok) {
        throw new Error(
          typeof data.detail === 'string'
            ? data.detail
            : 'Unable to load event documents.'
        )
      }

      setDocuments(
        Array.isArray(data.documents)
          ? data.documents
          : []
      )
    } catch (requestError) {
      console.error(requestError)

      setDocuments([])

      setError(
        requestError.message ||
          'Unable to load event documents.'
      )
    } finally {
      setDocumentsLoading(false)
    }
  }

  const viewDocument = (documentId) => {
    const fileUrl =
      `${API_URL}/documents/${documentId}/file`

    window.open(
      fileUrl,
      '_blank',
      'noopener,noreferrer'
    )
  }

  const handleUpload = async (event) => {
    event.preventDefault()

    if (!selectedEventId) {
      setError('Please select an event first.')
      return
    }

    if (!documentType) {
      setError('Please select a document type.')
      return
    }

    if (!selectedFile) {
      setError('Please choose a file to upload.')
      return
    }

    try {
      setUploading(true)
      setError('')
      setSuccess('')

      const formData = new FormData()

      formData.append(
        'event_id',
        selectedEventId
      )

      formData.append(
        'document_type',
        documentType
      )

      formData.append(
        'file',
        selectedFile
      )

      const response = await fetch(
        `${API_URL}/documents/upload`,
        {
          method: 'POST',
          body: formData,
        }
      )

      const data = await response.json()

      if (!response.ok) {
        let message =
          'Unable to upload document.'

        if (
          typeof data.detail === 'string'
        ) {
          message = data.detail
        }

        throw new Error(message)
      }

      setSuccess(
        'Document uploaded successfully.'
      )

      setDocumentType('')
      setSelectedFile(null)

      const fileInput =
        document.getElementById(
          'document-file'
        )

      if (fileInput) {
        fileInput.value = ''
      }

      await fetchDocuments(
        selectedEventId
      )
    } catch (requestError) {
      console.error(requestError)

      if (
        requestError instanceof TypeError
      ) {
        setError(
          'Unable to connect to the backend server. Make sure FastAPI is running.'
        )
      } else {
        setError(
          requestError.message ||
            'Something went wrong while uploading the document.'
        )
      }
    } finally {
      setUploading(false)
    }
  }

  const startReview = (document) => {
    setReviewingDocumentId(
      document.id
    )

    setReviewStatus(
      document.status || 'Pending'
    )

    setReviewRemarks(
      document.remarks || ''
    )

    setError('')
    setSuccess('')
  }

  const cancelReview = () => {
    setReviewingDocumentId(null)
    setReviewStatus('')
    setReviewRemarks('')
  }

  const saveReview = async (
    documentId
  ) => {
    if (!reviewStatus) {
      setError(
        'Please select a review status.'
      )
      return
    }

    try {
      setError('')
      setSuccess('')

      const response = await fetch(
        `${API_URL}/documents/${documentId}/status`,
        {
          method: 'PUT',
          headers: {
            'Content-Type':
              'application/json',
          },
          body: JSON.stringify({
            status: reviewStatus,
            remarks:
              reviewRemarks.trim() ||
              null,
          }),
        }
      )

      const data = await response.json()

      if (!response.ok) {
        throw new Error(
          typeof data.detail === 'string'
            ? data.detail
            : 'Unable to update document review.'
        )
      }

      setSuccess(
        'Document review updated successfully.'
      )

      cancelReview()

      await fetchDocuments(
        selectedEventId
      )
    } catch (requestError) {
      console.error(requestError)

      setError(
        requestError.message ||
          'Unable to update document review.'
      )
    }
  }

  const getSelectedEvent = () => {
    return events.find(
      (event) =>
        String(event.id) ===
        String(selectedEventId)
    )
  }

  const getStatusClass = (
    status
  ) => {
    if (status === 'Approved') {
      return 'document-status-approved'
    }

    if (status === 'Rejected') {
      return 'document-status-rejected'
    }

    if (
      status === 'Under Review'
    ) {
      return 'document-status-review'
    }

    if (
      status ===
      'Additional Information Required'
    ) {
      return 'document-status-info'
    }

    return 'document-status-pending'
  }

  const selectedEvent =
    getSelectedEvent()

  return (
    <div className="documents-page">
      <Navbar />

      <main className="documents-main">
        <section className="documents-heading">
          <div>
            <span>
              EVENT SAFETY DOCUMENTS
            </span>

            <h1>
              Document{' '}
              <strong>
                Management
              </strong>
            </h1>

            <p>
              {authority
                ? 'Open and review submitted event documents, then record the official review status and remarks.'
                : 'Upload required event documents, view their review status, and respond to official remarks.'}
            </p>
          </div>
        </section>

        {error && (
          <div className="documents-message documents-error">
            <span>!</span>

            <div>
              <strong>
                Something went wrong
              </strong>
              <p>{error}</p>
            </div>
          </div>
        )}

        {success && (
          <div className="documents-message documents-success">
            <span>✓</span>

            <div>
              <strong>
                Success
              </strong>
              <p>{success}</p>
            </div>
          </div>
        )}

        <section className="document-event-section">
          <div className="document-section-heading">
            <div className="document-section-number">
              01
            </div>

            <div>
              <h2>
                Select Event
              </h2>

              <p>
                Choose the registered event
                whose documents you want to
                manage.
              </p>
            </div>
          </div>

          {eventsLoading ? (
            <div className="documents-loading">
              Loading registered events...
            </div>
          ) : (
            <div className="document-event-select">
              <label htmlFor="document-event">
                Registered Event
              </label>

              <select
                id="document-event"
                value={selectedEventId}
                onChange={(event) => {
                  setSelectedEventId(
                    event.target.value
                  )

                  setSuccess('')
                  setError('')
                  cancelReview()
                }}
              >
                <option value="">
                  Select an event
                </option>

                {events.map(
                  (event) => (
                    <option
                      key={event.id}
                      value={event.id}
                    >
                      #{event.id} —{' '}
                      {event.event_name}
                    </option>
                  )
                )}
              </select>
            </div>
          )}

          {selectedEvent && (
            <div className="selected-event-summary">
              <div>
                <span>EVENT</span>

                <strong>
                  {selectedEvent.event_name}
                </strong>
              </div>

              <div>
                <span>EVENT ID</span>

                <strong>
                  #{selectedEvent.id}
                </strong>
              </div>

              <div>
                <span>LOCATION</span>

                <strong>
                  {selectedEvent.location}
                </strong>
              </div>
            </div>
          )}
        </section>

        {selectedEventId && (
          <>
            {!authority && (
              <section className="document-upload-section">
                <div className="document-section-heading">
                  <div className="document-section-number">
                    02
                  </div>

                  <div>
                    <h2>
                      Upload Document
                    </h2>

                    <p>
                      Add an event safety or
                      permission-related
                      document for official
                      review.
                    </p>
                  </div>
                </div>

                <form
                  className="document-upload-form"
                  onSubmit={handleUpload}
                >
                  <div className="document-form-group">
                    <label htmlFor="document-type">
                      Document Type
                    </label>

                    <select
                      id="document-type"
                      value={documentType}
                      onChange={(event) =>
                        setDocumentType(
                          event.target.value
                        )
                      }
                      required
                    >
                      <option value="">
                        Select document type
                      </option>

                      {DOCUMENT_TYPES.map(
                        (type) => (
                          <option
                            key={type}
                            value={type}
                          >
                            {type}
                          </option>
                        )
                      )}
                    </select>
                  </div>

                  <div className="document-form-group">
                    <label htmlFor="document-file">
                      Choose File
                    </label>

                    <input
                      id="document-file"
                      type="file"
                      onChange={(event) =>
                        setSelectedFile(
                          event.target.files?.[0] ||
                            null
                        )
                      }
                      required
                    />

                    {selectedFile && (
                      <small>
                        Selected:{' '}
                        {selectedFile.name}
                      </small>
                    )}
                  </div>

                  <button
                    className="document-upload-button"
                    type="submit"
                    disabled={uploading}
                  >
                    {uploading
                      ? 'Uploading...'
                      : 'Upload Document'}

                    <span>↑</span>
                  </button>
                </form>
              </section>
            )}

            <section className="documents-list-section">
              <div className="document-section-heading">
                <div className="document-section-number">
                  {authority
                    ? '02'
                    : '03'}
                </div>

                <div>
                  <h2>
                    Uploaded Documents
                  </h2>

                  <p>
                    {authority
                      ? 'Open each submitted document before recording the official review status.'
                      : 'View the documents attached to this event and their current review status.'}
                  </p>
                </div>
              </div>

              {documentsLoading ? (
                <div className="documents-loading">
                  Loading documents...
                </div>
              ) : documents.length ===
                0 ? (
                <div className="documents-empty">
                  <div>▤</div>

                  <h3>
                    No documents uploaded
                  </h3>

                  <p>
                    {authority
                      ? 'No documents have been submitted for this event yet.'
                      : 'Upload the first document for this event using the form above.'}
                  </p>
                </div>
              ) : (
                <div className="documents-grid">
                  {documents.map(
                    (document) => (
                      <article
                        className="document-card"
                        key={document.id}
                      >
                        <div className="document-card-top">
                          <div className="document-file-icon">
                            ▤
                          </div>

                          <span
                            className={`document-status ${getStatusClass(
                              document.status
                            )}`}
                          >
                            {document.status ||
                              'Pending'}
                          </span>
                        </div>

                        <div className="document-card-content">
                          <span className="document-type">
                            {
                              document.document_type
                            }
                          </span>

                          <h3>
                            {
                              document.document_name
                            }
                          </h3>

                          <p>
                            Document ID #
                            {document.id}
                          </p>
                        </div>

                        <div className="document-view-action">
                          <button
                            type="button"
                            className="document-view-button"
                            onClick={() =>
                              viewDocument(
                                document.id
                              )
                            }
                          >
                            View Document ↗
                          </button>
                        </div>

                        {document.remarks && (
                          <div className="document-remarks">
                            <span>
                              {authority
                                ? 'CURRENT REVIEW REMARKS'
                                : 'AUTHORITY REMARKS'}
                            </span>

                            <p>
                              {document.remarks}
                            </p>
                          </div>
                        )}

                        {authority &&
                        reviewingDocumentId ===
                          document.id ? (
                          <div className="document-review-form">
                            <div className="document-form-group">
                              <label>
                                Review Status
                              </label>

                              <select
                                value={reviewStatus}
                                onChange={(event) =>
                                  setReviewStatus(
                                    event.target.value
                                  )
                                }
                              >
                                {DOCUMENT_STATUSES.map(
                                  (status) => (
                                    <option
                                      key={status}
                                      value={status}
                                    >
                                      {status}
                                    </option>
                                  )
                                )}
                              </select>
                            </div>

                            <div className="document-form-group">
                              <label>
                                Remarks
                              </label>

                              <textarea
                                rows="3"
                                placeholder="Enter review remarks..."
                                value={reviewRemarks}
                                onChange={(event) =>
                                  setReviewRemarks(
                                    event.target.value
                                  )
                                }
                              />
                            </div>

                            <div className="document-review-actions">
                              <button
                                type="button"
                                className="document-cancel-button"
                                onClick={cancelReview}
                              >
                                Cancel
                              </button>

                              <button
                                type="button"
                                className="document-save-button"
                                onClick={() =>
                                  saveReview(
                                    document.id
                                  )
                                }
                              >
                                Save Review
                              </button>
                            </div>
                          </div>
                        ) : (
                          <div className="document-card-actions">
                            {authority ? (
                              <button
                                type="button"
                                className="document-review-button"
                                onClick={() =>
                                  startReview(
                                    document
                                  )
                                }
                              >
                                Review Document
                              </button>
                            ) : (
                              <span className="document-organizer-note">
                                Status changes are
                                made by the
                                reviewing authority.
                              </span>
                            )}
                          </div>
                        )}
                      </article>
                    )
                  )}
                </div>
              )}
            </section>
          </>
        )}

        <section className="documents-note">
          <span>i</span>

          <div>
            <strong>
              Authorized Review Support
            </strong>

            <p>
              CrowdGuard records document
              review status and remarks to
              support authorized officials.
              The system does not independently
              grant legal event permission.
            </p>
          </div>
        </section>
      </main>
    </div>
  )
}

export default Documents
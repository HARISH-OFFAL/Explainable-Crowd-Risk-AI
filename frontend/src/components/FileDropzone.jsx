function FileDropzone({ file, onChange, accept = '.pdf,.doc,.docx,.jpg,.jpeg,.png' }) {
  const chooseFile = (event) => onChange(event.target.files?.[0] || null)

  return (
    <label className="file-dropzone" htmlFor="document-file">
      <input id="document-file" type="file" accept={accept} onChange={chooseFile} />
      <span className="dropzone-icon">↑</span>
      <strong>{file ? file.name : 'Drag and drop your document here'}</strong>
      <small>{file ? `${(file.size / 1024 / 1024).toFixed(2)} MB · ${file.type || 'File'}` : 'or click to browse · PDF, DOCX, JPG, PNG'}</small>
      {file && <button type="button" className="dropzone-remove" onClick={(event) => { event.preventDefault(); onChange(null) }}>Remove file</button>}
    </label>
  )
}

export default FileDropzone

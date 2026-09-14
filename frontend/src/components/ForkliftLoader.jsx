function ForkliftLoader({ label = 'Preparing Flow Intelligence' }) {
  return <div className="forklift-loader" role="status" aria-label={label}>
    <div className="flow-ai-loader">
      <div className="flow-ai-grid" />
      <div className="flow-ai-orbit orbit-one" />
      <div className="flow-ai-orbit orbit-two" />
      <div className="flow-ai-core"><span>AI</span><i /></div>
      <div className="flow-ai-node node-one" /><div className="flow-ai-node node-two" /><div className="flow-ai-node node-three" />
      <div className="flow-ai-path path-one" /><div className="flow-ai-path path-two" /><div className="flow-ai-path path-three" />
      <div className="flow-ai-scan" />
    </div>
    <div className="forklift-loader-copy"><strong>{label}</strong><span>Mapping movement patterns and crowd flow...</span><div className="flow-ai-progress"><i /></div><small>Synchronizing zone intelligence</small></div>
  </div>
}

export default ForkliftLoader

function ForkliftLoader({ label = 'Preparing Flow Intelligence' }) {
  return <div className="forklift-loader" role="status" aria-label={label}>
    <div className="forklift-scene">
      <div className="forklift-floor" />
      <div className="forklift-conveyor" />
      <div className="forklift-box forklift-box-three" />
      <div className="forklift-box forklift-box-two" />
      <div className="forklift-box forklift-box-one" />
      <div className="forklift-box forklift-box-five" />
      <div className="forklift-box forklift-box-six" />
      <div className="forklift-box forklift-box-seven" />
      <div className="forklift-box forklift-moving-box" />
      <div className="forklift-lift">
        <img className="forklift-body" src="/figma/forklift.svg" alt="" />
        <img className="forklift-fork-vertical" src="/figma/fork-vertical.svg" alt="" />
        <img className="forklift-fork-bottom" src="/figma/fork-bottom.svg" alt="" />
      </div>
    </div>
    <div className="forklift-loader-copy"><strong>{label}</strong><span>Following the movement sequence…</span></div>
  </div>
}

export default ForkliftLoader

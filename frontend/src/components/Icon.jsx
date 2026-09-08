const PATHS = {
  plus: 'M12 5v14M5 12h14',
  list: 'M5 6h14M5 12h14M5 18h14',
  upload: 'M12 16V4m0 0L8 8m4-4 4 4M5 15v4h14v-4',
  check: 'm5 12 4 4L19 6',
}

function Icon({ name = 'list', size = 20 }) {
  return <svg className="ui-icon" width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true"><path d={PATHS[name] || PATHS.list} /></svg>
}

export default Icon

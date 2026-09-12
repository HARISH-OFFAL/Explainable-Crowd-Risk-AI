export const ROLES = {
  ORGANIZER: 'EVENT_ORGANIZER',
  AUTHORITY: 'AUTHORITY',
}

export const DEMO_CREDENTIALS = {
  [ROLES.ORGANIZER]: { email: 'organizer@crowdguard.demo', password: 'Organizer@123' },
  [ROLES.AUTHORITY]: { email: 'authority@crowdguard.demo', password: 'Authority@123' },
}

export const ROLE_LABELS = {
  [ROLES.ORGANIZER]: 'Event Organizer',
  [ROLES.AUTHORITY]: 'Authority',
}

export const PORTAL_SWITCH_PASSWORD = '1234'

// Demo-only authentication for academic project use. Replace with server-side authentication before production deployment.

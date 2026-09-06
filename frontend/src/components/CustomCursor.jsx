import { useEffect, useRef } from 'react'

function CustomCursor() {
  const cursorRef = useRef(null)

  useEffect(() => {
    const cursor = cursorRef.current
    const coarsePointer = window.matchMedia('(pointer: coarse)')

    if (!cursor || coarsePointer.matches) {
      return undefined
    }

    let frame = 0
    let nextPosition = { x: 0, y: 0 }

    const setInteractiveState = (target) => {
      const element = target instanceof Element ? target : null
      const isInput = Boolean(
        element?.closest('input, textarea, select, [contenteditable="true"]')
      )
      const isInteractive = Boolean(
        element?.closest('a, button, [role="button"], .glass-card')
      )

      document.body.classList.toggle('cg-custom-cursor-active', !isInput)
      cursor.classList.toggle('is-hidden', isInput)
      cursor.classList.toggle('is-interactive', !isInput && isInteractive)
    }

    const moveCursor = (event) => {
      nextPosition = { x: event.clientX, y: event.clientY }
      setInteractiveState(event.target)

      if (!frame) {
        frame = window.requestAnimationFrame(() => {
          cursor.style.transform = `translate3d(${nextPosition.x}px, ${nextPosition.y}px, 0)`
          frame = 0
        })
      }
    }

    const hideCursor = () => cursor.classList.add('is-hidden')
    const showCursor = () => cursor.classList.remove('is-hidden')

    document.body.classList.add('cg-custom-cursor-active')
    window.addEventListener('pointermove', moveCursor, { passive: true })
    window.addEventListener('pointerleave', hideCursor)
    window.addEventListener('pointerenter', showCursor)

    return () => {
      if (frame) window.cancelAnimationFrame(frame)
      document.body.classList.remove('cg-custom-cursor-active')
      window.removeEventListener('pointermove', moveCursor)
      window.removeEventListener('pointerleave', hideCursor)
      window.removeEventListener('pointerenter', showCursor)
    }
  }, [])

  return (
    <span ref={cursorRef} className="cg-cursor" aria-hidden="true">
      <span className="cg-cursor-dot" />
      <span className="cg-cursor-ring" />
    </span>
  )
}

export default CustomCursor

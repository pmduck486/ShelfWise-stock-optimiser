import { useEffect, useRef, useState } from 'react';
import { animate, useInView, useReducedMotion } from 'motion/react';

/** NumberTicker: counts from the previous value to the new one once visible. */
export function AnimatedNumber({
  value,
  format,
  className,
  duration = 1.1,
}: {
  value: number;
  format: (x: number) => string;
  className?: string;
  duration?: number;
}) {
  const ref = useRef<HTMLSpanElement>(null);
  const inView = useInView(ref, { once: true, margin: '-20px' });
  const reduce = useReducedMotion();
  const from = useRef<number | null>(null);
  const [shown, setShown] = useState(reduce ? value : value * 0);

  useEffect(() => {
    if (reduce) {
      setShown(value);
      from.current = value;
      return;
    }
    if (!inView) return;
    const start = from.current ?? 0;
    const controls = animate(start, value, {
      duration,
      ease: [0.16, 1, 0.3, 1],
      onUpdate: (v) => setShown(v),
    });
    from.current = value;
    // If animation frames are paused (background tab, print, headless capture), still land on the value.
    const settle = window.setTimeout(() => setShown(value), duration * 1000 + 400);
    return () => {
      controls.stop();
      window.clearTimeout(settle);
    };
  }, [value, inView, reduce, duration]);

  return (
    <span ref={ref} className={className}>
      <span aria-hidden>{format(shown)}</span>
      <span className="sr-only">{format(value)}</span>
    </span>
  );
}

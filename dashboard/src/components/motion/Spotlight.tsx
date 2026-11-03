import { useRef, type ReactNode } from 'react';
import { motion, useMotionTemplate, useMotionValue, useReducedMotion, useSpring } from 'motion/react';

/** A soft light that follows the pointer across a surface. */
export function Spotlight({ children, className = '', size = 420 }: { children: ReactNode; className?: string; size?: number }) {
  const ref = useRef<HTMLDivElement>(null);
  const reduce = useReducedMotion();
  const x = useSpring(useMotionValue(-999), { stiffness: 220, damping: 30 });
  const y = useSpring(useMotionValue(-999), { stiffness: 220, damping: 30 });
  const bg = useMotionTemplate`radial-gradient(${size}px circle at ${x}px ${y}px, color-mix(in oklab, var(--accent) 13%, transparent), transparent 70%)`;
  return (
    <div
      ref={ref}
      className={`relative overflow-hidden ${className}`}
      onPointerMove={(e) => {
        if (reduce || !ref.current || e.pointerType !== 'mouse') return;
        const r = ref.current.getBoundingClientRect();
        x.set(e.clientX - r.left);
        y.set(e.clientY - r.top);
      }}
    >
      {!reduce && <motion.div aria-hidden className="pointer-events-none absolute inset-0" style={{ background: bg }} />}
      <div className="relative">{children}</div>
    </div>
  );
}

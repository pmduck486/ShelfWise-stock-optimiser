import { AnimatePresence, motion, useReducedMotion } from 'motion/react';
import type { ReactNode } from 'react';

/** Shows one panel at a time; the swap slides in the direction of travel. */
export function TransitionPanel({
  active,
  direction,
  children,
  className,
}: {
  active: number;
  direction: 1 | -1;
  children: ReactNode[];
  className?: string;
}) {
  const reduce = useReducedMotion();
  const dist = reduce ? 0 : 36;
  return (
    <div className={className} style={{ position: 'relative' }}>
      <AnimatePresence initial={false} mode="wait">
        <motion.div
          key={active}
          initial={{ opacity: 0, x: direction * dist, filter: reduce ? 'none' : 'blur(4px)' }}
          animate={{ opacity: 1, x: 0, filter: 'blur(0px)' }}
          exit={{ opacity: 0, x: -direction * dist, filter: reduce ? 'none' : 'blur(4px)' }}
          transition={{ duration: reduce ? 0.01 : 0.3, ease: [0.16, 1, 0.3, 1] }}
        >
          {children[active]}
        </motion.div>
      </AnimatePresence>
    </div>
  );
}

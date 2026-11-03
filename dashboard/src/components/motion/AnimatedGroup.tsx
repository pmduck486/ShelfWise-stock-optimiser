import { motion, useReducedMotion, type Variants } from 'motion/react';
import type { ReactNode } from 'react';

/** Children (motion elements using `groupItem`) enter one after another. */
export function AnimatedGroup({
  children,
  className,
  stagger = 0.06,
  delay = 0,
}: {
  children: ReactNode;
  className?: string;
  stagger?: number;
  delay?: number;
}) {
  const reduce = useReducedMotion();
  const container: Variants = { hidden: {}, show: { transition: { staggerChildren: reduce ? 0 : stagger, delayChildren: delay } } };
  return (
    <motion.div className={className} initial="hidden" whileInView="show" viewport={{ once: true, margin: '-30px' }} variants={container}>
      {children}
    </motion.div>
  );
}

export const groupItem: Variants = {
  hidden: { opacity: 0, y: 14, filter: 'blur(4px)' },
  show: { opacity: 1, y: 0, filter: 'blur(0px)', transition: { duration: 0.5, ease: [0.16, 1, 0.3, 1] } },
};

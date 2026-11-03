import { motion, useReducedMotion, type Variants } from 'motion/react';

const container = (stagger: number, delay: number): Variants => ({
  hidden: {},
  show: { transition: { staggerChildren: stagger, delayChildren: delay } },
});
const piece: Variants = {
  hidden: { opacity: 0, y: '0.35em', filter: 'blur(6px)' },
  show: { opacity: 1, y: 0, filter: 'blur(0px)', transition: { duration: 0.55, ease: [0.16, 1, 0.3, 1] } },
};

/** Text that assembles word by word (fade + blur + rise). Adapted from the motion-primitives idea. */
export function TextEffect({
  children,
  as = 'p',
  className,
  delay = 0,
  stagger = 0.045,
}: {
  children: string;
  as?: 'p' | 'h1' | 'h2' | 'h3' | 'span';
  className?: string;
  delay?: number;
  stagger?: number;
}) {
  const reduce = useReducedMotion();
  const Tag = as;
  if (reduce) return <Tag className={className}>{children}</Tag>;
  const M = motion[as];
  const parts = children.split(/(\s+)/);
  return (
    <M className={className} initial="hidden" animate="show" variants={container(stagger, delay)} aria-label={children}>
      {parts.map((p, i) =>
        /^\s+$/.test(p) ? (
          <span key={i} aria-hidden>
            {p}
          </span>
        ) : (
          <motion.span key={i} variants={piece} aria-hidden style={{ display: 'inline-block', whiteSpace: 'pre' }}>
            {p}
          </motion.span>
        ),
      )}
    </M>
  );
}

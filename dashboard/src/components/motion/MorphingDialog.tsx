import { useEffect, useRef, type ReactNode } from 'react';
import { createPortal } from 'react-dom';
import { AnimatePresence, motion, useReducedMotion } from 'motion/react';
import { X } from 'lucide-react';

/**
 * MorphingDialog: the trigger (any motion element with the same `layoutId`) grows into this
 * dialog and shrinks back on close. Escape and the backdrop close it, focus is kept inside
 * and returned to the trigger afterwards.
 */
export function MorphingDialog({
  open,
  onClose,
  layoutId,
  label,
  children,
}: {
  open: boolean;
  onClose: () => void;
  layoutId: string;
  label: string;
  children: ReactNode;
}) {
  const reduce = useReducedMotion();
  const panel = useRef<HTMLDivElement>(null);
  const returnTo = useRef<HTMLElement | null>(null);

  useEffect(() => {
    if (!open) return;
    returnTo.current = document.activeElement as HTMLElement | null;
    const prev = document.body.style.overflow;
    document.body.style.overflow = 'hidden';
    const t = window.setTimeout(() => panel.current?.querySelector<HTMLElement>('[data-autofocus]')?.focus(), 60);
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') {
        e.preventDefault();
        onClose();
      } else if (e.key === 'Tab' && panel.current) {
        const f = panel.current.querySelectorAll<HTMLElement>(
          'button:not([disabled]), [href], input, select, textarea, [tabindex]:not([tabindex="-1"])',
        );
        if (!f.length) return;
        const first = f[0];
        const last = f[f.length - 1];
        if (e.shiftKey && document.activeElement === first) {
          e.preventDefault();
          last.focus();
        } else if (!e.shiftKey && document.activeElement === last) {
          e.preventDefault();
          first.focus();
        }
      }
    };
    document.addEventListener('keydown', onKey);
    return () => {
      document.body.style.overflow = prev;
      document.removeEventListener('keydown', onKey);
      window.clearTimeout(t);
      returnTo.current?.focus?.();
    };
  }, [open, onClose]);

  return createPortal(
    <AnimatePresence>
      {open && (
        <div className="fixed inset-0 z-50 flex items-stretch justify-center sm:items-center sm:p-6">
          <motion.div
            className="absolute inset-0 bg-black/45 backdrop-blur-[2px]"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            transition={{ duration: 0.25 }}
            onClick={onClose}
          />
          <motion.div
            ref={panel}
            role="dialog"
            aria-modal="true"
            aria-label={label}
            layoutId={reduce ? undefined : layoutId}
            className="relative flex max-h-full w-full max-w-6xl flex-col overflow-hidden bg-surface shadow-lift sm:max-h-[92vh] sm:rounded-2xl"
            style={{ borderRadius: 16 }}
            initial={reduce ? { opacity: 0 } : undefined}
            animate={reduce ? { opacity: 1 } : undefined}
            exit={reduce ? { opacity: 0 } : undefined}
            transition={{ type: 'spring', bounce: 0.08, duration: 0.5 }}
          >
            <button
              type="button"
              onClick={onClose}
              aria-label="Close day detail"
              className="absolute right-3 top-3 z-10 grid size-9 place-items-center rounded-full bg-surface-2 text-ink-2 transition-colors hover:bg-surface-3 hover:text-ink"
            >
              <X size={18} aria-hidden />
            </button>
            <motion.div
              className="overflow-y-auto overscroll-contain"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1, transition: { delay: reduce ? 0 : 0.18, duration: 0.25 } }}
              exit={{ opacity: 0, transition: { duration: 0.08 } }}
            >
              {children}
            </motion.div>
          </motion.div>
        </div>
      )}
    </AnimatePresence>,
    document.body,
  );
}

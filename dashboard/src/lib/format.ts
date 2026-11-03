/* Number formatting. Money is in M5 shelf-price units (US dollars from Walmart's sell_prices;
   the project notes write them as Rs), so it is shown with "$". */

const int = new Intl.NumberFormat('en-US', { maximumFractionDigits: 0 });

export const fmtInt = (x: number | null | undefined) => (x == null || !Number.isFinite(x) ? '–' : int.format(Math.round(x)));

export const fmtNum = (x: number | null | undefined, d = 1) =>
  x == null || !Number.isFinite(x)
    ? '–'
    : new Intl.NumberFormat('en-US', { minimumFractionDigits: d, maximumFractionDigits: d }).format(x);

export const fmtMoney = (x: number | null | undefined, d = 0) =>
  x == null || !Number.isFinite(x)
    ? '–'
    : (x < 0 ? '−$' : '$') + new Intl.NumberFormat('en-US', { minimumFractionDigits: d, maximumFractionDigits: d }).format(Math.abs(x));

export const fmtPct = (share: number | null | undefined, d = 2) =>
  share == null || !Number.isFinite(share) ? '–' : `${fmtNum(share * 100, d)}%`;

export const fmtSigned = (x: number, d = 1, unit = '%') => `${x > 0 ? '+' : x < 0 ? '−' : ''}${fmtNum(Math.abs(x), d)}${unit}`;

const MONTHS = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];
const DAYS = ['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat'];

export function parseISO(s: string) {
  const [y, m, d] = s.split('-').map(Number);
  return { y, m, d, dow: new Date(Date.UTC(y, m - 1, d)).getUTCDay() };
}

export const fmtDate = (s: string, withDow = false) => {
  const p = parseISO(s);
  return `${withDow ? DAYS[p.dow] + ' ' : ''}${p.d} ${MONTHS[p.m - 1]}${withDow ? '' : ''} ${p.y}`;
};
export const fmtDay = (s: string) => {
  const p = parseISO(s);
  return `${p.d} ${MONTHS[p.m - 1]}`;
};
export const dowName = (s: string) => DAYS[parseISO(s).dow];
export const DOW_LABELS = DAYS;

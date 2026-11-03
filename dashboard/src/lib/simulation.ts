// Mathematical interpolation and continuous simulation model for Shelfwise What-If Limits
// Grounded in empirical solver replay points on the Walmart M5 test window

export interface SimMetrics {
  value: number;
  shadowPrice: number;
  bindingDays: number;
  fill: number;
  lost: number;
  waste: number;
  total: number;
  netCost: number;
  severity: 'none' | 'low' | 'medium' | 'high';
}

function hermite(points: [number, number][], x: number): number {
  if (x <= points[0][0]) {
    const [x0, y0] = points[0];
    const [x1, y1] = points[1];
    const slope = (y1 - y0) / (x1 - x0);
    return Math.max(0, y0 + slope * (x - x0));
  }
  if (x >= points[points.length - 1][0]) {
    return points[points.length - 1][1];
  }
  for (let i = 0; i < points.length - 1; i++) {
    const [x0, y0] = points[i];
    const [x1, y1] = points[i + 1];
    if (x >= x0 && x <= x1) {
      const t = (x - x0) / (x1 - x0);
      const s = t * t * (3 - 2 * t);
      return y0 + s * (y1 - y0);
    }
  }
  return points[points.length - 1][1];
}

// 1. Shelf Space Factor (0.10 to 1.00)
// Measured empirical points: 0.2, 0.3, 0.4, 0.5, 0.7, 1.0 (all layers enabled)
const SPACE_SHADOW: [number, number][] = [
  [0.10, 38.5],
  [0.20, 19.2899],
  [0.30, 4.6593],
  [0.40, 0.4432],
  [0.50, 0.0214],
  [0.70, 0.0],
  [1.00, 0.0],
];
const SPACE_DAYS: [number, number][] = [
  [0.10, 550],
  [0.20, 540],
  [0.30, 477],
  [0.40, 107],
  [0.50, 9],
  [0.70, 0],
  [1.00, 0],
];
const SPACE_FILL: [number, number][] = [
  [0.10, 0.50],
  [0.20, 0.716999],
  [0.30, 0.882231],
  [0.40, 0.938315],
  [0.50, 0.949089],
  [0.70, 0.949622],
  [1.00, 0.949622],
];
const SPACE_LOST: [number, number][] = [
  [0.10, 240000],
  [0.20, 144520.5],
  [0.30, 60141.1],
  [0.40, 31500.5],
  [0.50, 25998.6],
  [0.70, 25726.6],
  [1.00, 25726.6],
];
const SPACE_TOTAL: [number, number][] = [
  [0.10, 1150000],
  [0.20, 993338.5],
  [0.30, 943011.7],
  [0.40, 933635.6],
  [0.50, 932206.1],
  [0.70, 932093.7],
  [1.00, 932093.7],
];
const SPACE_WASTE: [number, number][] = [
  [0.10, 2500],
  [0.20, 4575.1],
  [0.30, 5766.9],
  [0.40, 6848.8],
  [0.50, 7015.4],
  [0.70, 7015.4],
  [1.00, 7015.4],
];

// 2. Daily Budget Factor (0.10 to 1.25)
// Measured empirical points: 0.3, 0.45, 0.6, 0.75, 1.25 (all layers enabled)
const BUDGET_SHADOW: [number, number][] = [
  [0.10, 35.0],
  [0.30, 18.8984],
  [0.45, 3.1701],
  [0.60, 0.2086],
  [0.75, 0.0193],
  [1.25, 0.0],
];
const BUDGET_DAYS: [number, number][] = [
  [0.10, 550],
  [0.30, 550],
  [0.45, 198],
  [0.60, 15],
  [0.75, 6],
  [1.25, 0],
];
const BUDGET_FILL: [number, number][] = [
  [0.10, 0.35],
  [0.30, 0.642659],
  [0.45, 0.934618],
  [0.60, 0.948774],
  [0.75, 0.949523],
  [1.25, 0.949622],
];
const BUDGET_LOST: [number, number][] = [
  [0.10, 290000],
  [0.30, 182483.5],
  [0.45, 33388.6],
  [0.60, 26159.6],
  [0.75, 25777.1],
  [1.25, 25726.6],
];
const BUDGET_TOTAL: [number, number][] = [
  [0.10, 1350000],
  [0.30, 1073729.5],
  [0.45, 940102.9],
  [0.60, 932501.5],
  [0.75, 932099.1],
  [1.25, 932093.7],
];
const BUDGET_WASTE: [number, number][] = [
  [0.10, 1500],
  [0.30, 3500.1],
  [0.45, 6412.1],
  [0.60, 7002.3],
  [0.75, 7001.7],
  [1.25, 7015.4],
];

// 3. Truck Capacity in Cartons (10 to 400 cases)
// Measured empirical points: 25, 60, 100, 200, 400 (all layers enabled)
const TRUCK_SHADOW: [number, number][] = [
  [10, 0.08],
  [25, 0.0324],
  [60, 0.0065],
  [100, 0.0],
  [200, 0.0],
  [400, 0.0],
];
const TRUCK_DAYS: [number, number][] = [
  [10, 20],
  [25, 5],
  [60, 1],
  [100, 0],
  [200, 0],
  [400, 0],
];
const TRUCK_FILL: [number, number][] = [
  [10, 0.945],
  [25, 0.948723],
  [60, 0.949234],
  [100, 0.949324],
  [200, 0.949622],
  [400, 0.949622],
];
const TRUCK_LOST: [number, number][] = [
  [10, 28000],
  [25, 26185.9],
  [60, 25924.7],
  [100, 25878.6],
  [200, 25726.6],
  [400, 25726.6],
];
const TRUCK_TOTAL: [number, number][] = [
  [10, 1080000],
  [25, 984149.8],
  [60, 945812.2],
  [100, 935355.2],
  [200, 932173.7],
  [400, 932093.7],
];
const TRUCK_WASTE: [number, number][] = [
  [10, 6800],
  [25, 7003.6],
  [60, 6985.5],
  [100, 7003.4],
  [200, 7015.4],
  [400, 7015.4],
];

export function simulateParameter(param: string, value: number): SimMetrics {
  let shadowPrice = 0;
  let bindingDays = 0;
  let fill = 0.948678;
  let lost = 26208.4;
  let waste = 8042.9;
  let total = 934087.7;

  if (param === 'analysis.store_ilp.space_factor') {
    const v = Math.max(0.10, Math.min(1.00, value));
    shadowPrice = hermite(SPACE_SHADOW, v);
    bindingDays = Math.round(hermite(SPACE_DAYS, v));
    fill = hermite(SPACE_FILL, v);
    lost = hermite(SPACE_LOST, v);
    total = hermite(SPACE_TOTAL, v);
    waste = hermite(SPACE_WASTE, v);
  } else if (param === 'analysis.store_ilp.budget_factor') {
    const v = Math.max(0.10, Math.min(1.25, value));
    shadowPrice = hermite(BUDGET_SHADOW, v);
    bindingDays = Math.round(hermite(BUDGET_DAYS, v));
    fill = hermite(BUDGET_FILL, v);
    lost = hermite(BUDGET_LOST, v);
    total = hermite(BUDGET_TOTAL, v);
    waste = hermite(BUDGET_WASTE, v);
  } else if (param === 'order_model.truck_capacity_cases') {
    const v = Math.max(10, Math.min(400, value));
    shadowPrice = hermite(TRUCK_SHADOW, v);
    bindingDays = Math.round(hermite(TRUCK_DAYS, v));
    fill = hermite(TRUCK_FILL, v);
    lost = hermite(TRUCK_LOST, v);
    total = hermite(TRUCK_TOTAL, v);
    waste = hermite(TRUCK_WASTE, v);
  } else {
    // Penalty share (1.0, 1.5, 2.0)
    if (value <= 1.0) {
      fill = 0.923437;
      lost = 39098.6;
      waste = 5498.0;
      total = 908493.5;
    } else if (value >= 2.0) {
      fill = 0.959939;
      lost = 20458.2;
      waste = 10111.5;
      total = 949809.6;
    }
  }

  // Determine severity
  let severity: 'none' | 'low' | 'medium' | 'high' = 'none';
  if (bindingDays >= 120) {
    severity = 'high';
  } else if (bindingDays >= 20) {
    severity = 'medium';
  } else if (bindingDays > 0) {
    severity = 'low';
  }

  const netCost = total - lost * 1.5 * 2.8; // approximate lost sale penalty share

  return {
    value,
    shadowPrice,
    bindingDays,
    fill,
    lost,
    waste,
    total,
    netCost,
    severity,
  };
}

export function generateCurvePoints(param: string, min: number, max: number, steps = 35): [number, number, number][] {
  // Returns [x, fillPct, totalCost]
  const pts: [number, number, number][] = [];
  const delta = (max - min) / (steps - 1);
  for (let i = 0; i < steps; i++) {
    const x = +(min + i * delta).toFixed(4);
    const sim = simulateParameter(param, x);
    pts.push([x, +(sim.fill * 100).toFixed(2), Math.round(sim.total)]);
  }
  return pts;
}

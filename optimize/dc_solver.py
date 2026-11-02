"""Exact block decomposition under the fixed-home-DC network boundary."""

from concurrent.futures import ThreadPoolExecutor
import math
from pathlib import Path
from time import perf_counter

import pulp

from .cbc import cbc_solver


def solve_dc_blocks(model,constraints,owner,dcs,settings,start_values,log_dir):
    tolerance=settings['feasibility_tolerance']
    reserve=settings['parallel_time_reserve_s']
    if not 0 <= reserve < settings['time_limit_s']:
        raise ValueError('Parallel time reserve must be below the total solve limit')
    limit=settings['time_limit_s']-reserve
    blocks={d:pulp.LpProblem('RetailOps_'+str(d),pulp.LpMinimize) for d in dcs}
    assigned={d:[] for d in dcs}
    for key,constraint in constraints.items():
        owners={owner[v] for v in constraint.keys()}
        if len(owners)>1:
            raise ValueError('Cannot split a cross-DC constraint: '+str(key))
        d=next(iter(owners)) if owners else dcs[0]
        blocks[d].addConstraint(constraint,name=constraint.name)
        assigned[d].append(constraint)
    for index,d in enumerate(dcs):
        objective=pulp.lpSum(coef*v for v,coef in model.objective.items() if owner[v]==d)
        if index==0:
            objective+=model.objective.constant
        blocks[d].setObjective(objective)

    def feasible(problem):
        for v in problem.variables():
            value=v.varValue
            if value is None or not math.isfinite(value):return False
            if v.lowBound is not None and value<v.lowBound-tolerance:return False
            if v.upBound is not None and value>v.upBound+tolerance:return False
            if v.isInteger() and abs(value-round(value))>tolerance:return False
        for c in problem.constraints.values():
            value=c.value()
            if value is None:return False
            if c.sense==pulp.LpConstraintEQ and abs(value)>tolerance:return False
            if c.sense==pulp.LpConstraintLE and value>tolerance:return False
            if c.sense==pulp.LpConstraintGE and value<-tolerance:return False
        return True

    def solve_one(entry):
        index,d=entry;problem=blocks[d]
        log_path=Path(log_dir)/f'dc_{index}.log'
        warm=start_values is not None
        initial={v:start_values[v] for v in problem.variables()} if warm else None
        initial_objective=float(pulp.value(problem.objective)) if warm else None
        solver=cbc_solver(msg=bool(settings['log']),timeLimit=limit,gapRel=settings['mip_gap'],
                          logPath=str(log_path),warmStart=warm)
        if not solver.available():raise RuntimeError('Bundled CBC unavailable')
        started=perf_counter();error=None
        try:problem.solve(solver)
        except pulp.PulpSolverError as exc:error=str(exc)
        elapsed=perf_counter()-started
        log=log_path.read_text(errors='replace') if log_path.exists() else ''
        valid=not error and problem.sol_status in (pulp.LpSolutionOptimal,pulp.LpSolutionIntegerFeasible) and feasible(problem)
        cbc_objective=float(pulp.value(problem.objective)) if valid else None
        fallback=False
        if initial is not None and settings.get('start_plan_fallback',False) and (
            not valid or initial_objective<cbc_objective-tolerance):
            for v,value in initial.items():v.varValue=value
            valid=feasible(problem);fallback=bool(valid)
        timeout='time limit' in log.lower() or 'stopped on time' in log.lower()
        if fallback:status='TimeLimit-Feasible' if timeout else 'Fallback-Feasible'
        elif valid:status=('TimeLimit-Feasible' if timeout else 'GapSatisfied-Feasible' if 'gap tolerance' in log.lower()
                           else 'Optimal' if problem.sol_status==pulp.LpSolutionOptimal else 'Feasible')
        else:status='TimeLimit-NoIncumbent' if timeout else 'NoValidatedIncumbent'
        import re
        bound_match=re.search(r'Lower bound:\s*([-+\deE.]+)',log)
        objective=float(pulp.value(problem.objective)) if valid else None
        bound=float(bound_match[1]) if bound_match else objective if status=='Optimal' else None
        return dict(dc_id=d,status=status,validated=bool(valid),used_start_fallback=fallback,
                    objective=objective,cbc_objective=cbc_objective,best_bound=bound,
                    solve_seconds=elapsed,time_limit_s=limit,error=error,log=log)

    started=perf_counter()
    with ThreadPoolExecutor(max_workers=len(dcs)) as pool:
        results=list(pool.map(solve_one,enumerate(dcs)))
    return results,perf_counter()-started

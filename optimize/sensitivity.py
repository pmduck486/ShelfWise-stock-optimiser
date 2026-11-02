"""Conditional fixed-integer LP handoff to L3; never overwrite the MILP."""

from copy import deepcopy
import math

import pulp


def fixed_integer_lp(plan, cfg):
    if not plan.stats.get('incumbent_validated') or plan.model is None:
        raise ValueError('Sensitivity requires a validated in-memory Plan')
    model, constraints = deepcopy((plan.model, plan.constraints))
    tolerance=cfg['order_model']['solver']['feasibility_tolerance']
    fixed=0
    for variable in model.variables():
        if variable.isInteger():
            value=variable.varValue
            if value is None or not math.isfinite(value) or abs(value-round(value))>tolerance:
                raise ValueError('Cannot fix an invalid integer incumbent')
            value=round(value)
            variable.lowBound=variable.upBound=value
            variable.cat=pulp.LpContinuous
            fixed+=1
    model.solve(pulp.PULP_CBC_CMD(msg=False,mip=False,
                timeLimit=cfg['order_model']['solver']['time_limit_s']))
    if model.status!=pulp.LpStatusOptimal:
        raise RuntimeError('Fixed-integer LP did not solve optimally')
    if any(constraint.pi is None or not math.isfinite(constraint.pi) for constraint in constraints.values()):
        raise RuntimeError('Fixed-integer LP has unavailable duals')
    return model,constraints,dict(fixed_integer_count=fixed,objective=float(pulp.value(model.objective)),
                                 interpretation='Planning-score sensitivity to each stored constraint RHS conditional on fixed incumbent integers; inspect sense/sign before translating to capacity; not cash savings or a MILP dual.')

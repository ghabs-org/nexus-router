"""Compare caller-evaluated, paired outcomes without inventing quality or tariffs."""
import math

def compare_outcomes(records):
    pairs={}; basis=None
    for record in records:
        case=record.get('case_id'); strategy=record.get('strategy')
        if not isinstance(case,str) or not case or strategy not in {'baseline','routed'}:
            raise ValueError('Each outcome requires case_id and baseline/routed strategy')
        for key in ('cost_usd','quality_score','latency_ms'):
            value=record.get(key)
            if type(value) not in {int,float} or not math.isfinite(value) or value<0:
                raise ValueError(f'Invalid outcome {key}')
        if record['quality_score']>1 or type(record.get('success')) is not bool:
            raise ValueError('Quality must be within 0..1 and success must be boolean')
        if record.get('cost_basis') not in {'estimated','billed'}:
            raise ValueError('Declare estimated or billed cost_basis')
        if basis is not None and basis!=record['cost_basis']:
            raise ValueError('Do not mix estimated and billed costs')
        basis=record['cost_basis']
        pair=pairs.setdefault(case,{})
        if strategy in pair:raise ValueError('Duplicate case/strategy outcome')
        pair[strategy]=record
    if not pairs or any(set(pair)!={'baseline','routed'} for pair in pairs.values()):
        raise ValueError('Every case requires both strategies')
    summary={}
    for strategy in ('baseline','routed'):
        rows=[pair[strategy] for pair in pairs.values()]
        successes=sum(row['success'] for row in rows)
        cost=sum(row['cost_usd'] for row in rows)
        summary[strategy]={'total_cost_usd':cost,'success_rate':successes/len(rows),
                           'mean_quality_score':sum(row['quality_score'] for row in rows)/len(rows),
                           'mean_latency_ms':sum(row['latency_ms'] for row in rows)/len(rows),
                           'cost_per_successful_task_usd':cost/successes if successes else None}
    baseline,routed=summary['baseline'],summary['routed']
    return {'mode':'provided_paired_outcomes','provenance':'Caller-supplied evaluations; not independently verified',
            'cases':len(pairs),'cost_basis':basis,**summary,
            'cost_reduction_fraction':1-routed['total_cost_usd']/baseline['total_cost_usd'] if baseline['total_cost_usd'] else None,
            'quality_delta':routed['mean_quality_score']-baseline['mean_quality_score'],
            'success_rate_delta':routed['success_rate']-baseline['success_rate']}

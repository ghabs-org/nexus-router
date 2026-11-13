"""Opt-in economic ordering within an approved profile; never moves existing pins."""
import hashlib
import json
import math
import time
from dataclasses import replace
from contextlib import closing
from . import db


def fingerprint(body, choice, principal, wire, provider=None):
    prefix=[body.get('system',''),body.get('instructions',''),body.get('tools',[])]
    encoded=json.dumps(prefix,sort_keys=True,separators=(',',':')).encode()
    key=hashlib.sha256(json.dumps([principal,wire,provider,choice,encoded.hex()],sort_keys=True).encode()).hexdigest()
    return key,len(encoded)//4


def order(result, body, policy, wire, pinned):
    cfg=policy.config.get('economics',{})
    if not cfg.get('enabled',False) or pinned or result.profile=='explicit':return result
    estimates=[];now=time.time()
    input_estimate=max(1,len(json.dumps(body,ensure_ascii=False).encode())//3)
    with closing(db._connect()) as conn:
        for choice in result.choices:
            model=policy.models[choice['target']];prices=model.get('pricing')
            quality=choice.get('quality_pass_rate',model.get('quality_pass_rate'))
            if not prices or type(quality) not in {int,float} or not math.isfinite(quality) or not 0<quality<=1:
                return replace(result,explanation={**result.explanation,'economics':{'applied':False,'reason':'missing_tariff_or_evaluated_quality'}})
            key,prefix_tokens=fingerprint(body,choice,result.principal,wire,policy.config['providers'][choice['target'].split(':',1)[0]])
            row=conn.execute('SELECT cached_tokens,observed_at FROM gateway_cache_hints WHERE cache_key=?',(key,)).fetchone()
            cached=min(input_estimate,prefix_tokens,row['cached_tokens']) if row and 0<=now-row['observed_at']<=cfg.get('hint_ttl_seconds',60) else 0
            output=min(body.get('max_tokens',16384),model['max_output_tokens'],cfg.get('expected_output_tokens',512))
            # A cold request may write cache entries at the most expensive tariff.
            cold_rate=max(prices[k] for k in ('input','cache_write_5m','cache_write_1h'))
            cost=((input_estimate-cached)*cold_rate+cached*prices['cache_read']+output*prices['output'])/1_000_000
            if not math.isfinite(cost/quality):
                return replace(result,explanation={**result.explanation,'economics':{'applied':False,'reason':'invalid_cost_estimate'}})
            estimates.append({'target':choice['target'],'estimated_cost_usd':cost,'expected_cost_per_pass_usd':cost/quality,
                              'cache_tokens_hint':cached,'cache_key':key,'quality_pass_rate':quality})
    quality_floor=max(cfg.get('quality_floor',.9),estimates[0]['quality_pass_rate']-cfg.get('max_quality_drop',0))
    for row in estimates:row['eligible']=row['quality_pass_rate']>=quality_floor
    ranked=sorted([row for row in estimates if row['eligible']],key=lambda row:row['expected_cost_per_pass_usd'])
    if not ranked:
        from .gateway_policy import PolicyError
        raise PolicyError('No profile candidate meets the configured economic quality floor',503)
    approved=tuple(choice for choice in result.choices if next(row['eligible'] for row in estimates if row['target']==choice['target']))
    margin=cfg.get('minimum_saving_fraction',.1)
    baseline=estimates[0]['expected_cost_per_pass_usd']
    apply=not estimates[0]['eligible'] or (baseline>0 and ranked[0]['expected_cost_per_pass_usd']<=baseline*(1-margin))
    choices=tuple(sorted(approved,key=lambda choice:next((not row['eligible'],row['expected_cost_per_pass_usd']) for row in estimates if row['target']==choice['target']))) if apply else approved
    return replace(result,choices=choices,explanation={**result.explanation,'economics':{
        'applied':apply,'reason':('quality_floor_replacement' if not estimates[0]['eligible'] else 'estimated_saving') if apply else 'below_saving_margin','estimates':estimates,
        'input_tokens_estimate':input_estimate,'prediction_only':True}})


def observe(result,target,usage):
    if result.explanation.get('pin_reused'):return
    economic=result.explanation.get('economics',{})
    row=next((r for r in economic.get('estimates',[]) if r['target']==target.label),None) if target else None
    if row is None:return
    cached=usage.get('cache_read_input_tokens')
    if type(cached) is not int or cached<0:return
    with closing(db._connect()) as conn,conn:
        conn.execute('INSERT INTO gateway_cache_hints VALUES (?,?,?) ON CONFLICT(cache_key) DO UPDATE SET cached_tokens=excluded.cached_tokens,observed_at=excluded.observed_at',
                     (row['cache_key'],cached,time.time()))
        conn.execute('DELETE FROM gateway_cache_hints WHERE observed_at<?',(time.time()-3600,))

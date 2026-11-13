#!/usr/bin/env python3
"""Offline routing regression benchmark. Does not measure answer quality or spend."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys
import time
import yaml
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.gateway_policy import GatewayPolicy

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--config',type=Path,default=ROOT/'src/gateway.example.yaml')
    parser.add_argument('--cases',type=Path,default=ROOT/'tests/fixtures/gateway_tasks.jsonl')
    parser.add_argument('--outcomes',type=Path,help='Compare paired caller-evaluated baseline/routed JSONL outcomes')
    args=parser.parse_args()
    if args.outcomes:
        from src.gateway_benchmark import compare_outcomes
        records=[json.loads(line) for line in args.outcomes.read_text().splitlines() if line.strip()]
        print(json.dumps(compare_outcomes(records),indent=2))
        return 0
    config=yaml.safe_load(args.config.read_text())
    # Only syntax/capability validation is needed. No provider or credential is contacted.
    for provider in config['providers'].values():
        if provider.get('transport')=='vertex':
            provider.setdefault('project','offline-benchmark')
            provider.setdefault('region','global')
            provider.setdefault('base_url','https://aiplatform.googleapis.com')
    policy=GatewayPolicy(config)
    failures=[];sources=Counter();count=0;correct=0
    start=time.perf_counter()
    for line in args.cases.read_text().splitlines():
        if not line.strip():continue
        case=json.loads(line)
        if case['profile'] not in policy.profiles:raise ValueError('Unknown expected profile')
        result=policy.classify({'model':policy.alias,'messages':[{'role':'user','content':case['task']}]},{},'offline')
        count+=1; correct+=result.profile==case['profile'];sources[result.explanation['source']]+=1
        if result.profile!=case['profile']:
            failures.append({'id':case['id'],'expected':case['profile'],'actual':result.profile})
    if not count:raise ValueError('Benchmark contains no cases')
    print(json.dumps({'mode':'offline_synthetic_routing','cases':count,'correct':correct,
                      'profile_accuracy':correct/count,'sources':dict(sources),'failures':failures,
                      'elapsed_ms':round((time.perf_counter()-start)*1000,2),
                      'answer_quality_measured':False,'cost_savings_measured':False},indent=2))
    return int(bool(failures))
if __name__=='__main__':raise SystemExit(main())

import copy
import threading
from concurrent.futures import ThreadPoolExecutor
import time
import pytest
from test_gateway import harness,payload
from src import db,gateway
from src.gateway_budget import BudgetLedger
from src.gateway_policy import PolicyError


def classification(server,text='Summarize this report'):
    return server.gateway.classification(payload(text),{},'test-principal')

def test_reservation_settlement_and_tenant_status(harness):
    client,server,upstream,config=harness
    server.gateway_auth.clients[0][2].update(max_cost_usd_24h=10.)
    response=client.post('/v1/messages',json=payload())
    assert response.status_code==200,response.text
    budget=client.get('/v1/nexus/status').json()['budget']
    assert budget['held_microusd']==207
    assert budget['reservations'][0]['state']=='settled'
    other=client.get('/v1/nexus/status',headers={'Authorization':'Bearer other-secret'}).json()['budget']
    assert other['reservations']==[]

def test_insufficient_funds_and_missing_tariffs_never_call_upstream(harness):
    client,server,upstream,config=harness
    server.gateway_auth.clients[0][2].update(max_cost_usd_24h=.001)
    assert client.post('/v1/messages',json=payload()).status_code==402
    assert not upstream.requests
    server.gateway_auth.clients[0][2].update(max_cost_usd_24h=10)
    config['models']['vertex:claude-opus-4-6'].pop('pricing')
    assert client.post('/v1/messages',json=payload()).status_code==400
    assert not upstream.requests

def test_cross_instance_admission_is_atomic(harness):
    client,server,upstream,config=harness
    ledger=BudgetLedger()
    first=classification(server)
    # A standard request reserves two fallbacks at 1.26144 USD each.
    ledger.reserve(first,{'max_cost_usd_24h':3},config,payload())
    def admit(_):
        try:
            BudgetLedger().reserve(classification(server),{'max_cost_usd_24h':3},config,payload())
            return True
        except PolicyError:return False
    with ThreadPoolExecutor(max_workers=4) as pool:
        assert not any(pool.map(admit,range(4)))
    ledger.finish(first.request_id)
    assert ledger.status(first.principal)['held_microusd']==0

def test_failed_or_crashed_request_holds_funds_and_can_be_reconciled(harness):
    client,server,upstream,config=harness
    item=classification(server)
    ledger=BudgetLedger();ledger.reserve(item,{'max_task_cost_usd':3},config,payload())
    ledger.attempt(item.request_id)
    with pytest.raises(ValueError,match='owner is alive'):
        ledger.reconcile(item.request_id,0,'premature release')
    # A different ledger instance sees the reservation after a simulated restart.
    before=BudgetLedger().status(item.principal)
    assert before['reservations'][0]['state']=='reserved'
    ledger.finish(item.request_id)
    assert ledger.status(item.principal)['reservations'][0]['state']=='uncertain'
    with db._connect() as conn:
        conn.execute('UPDATE gateway_budget_reservations SET created_at=?',(time.time()-900000,))
    with pytest.raises(PolicyError):
        ledger.reserve(classification(server),{'max_cost_usd_24h':3},config,payload())
    ledger.reconcile(item.request_id,.1,'Reviewed provider billing')
    assert ledger.status(item.principal)['held_microusd']==100000
    with db._connect() as conn:
        assert conn.execute('SELECT COUNT(*) FROM gateway_budget_audit').fetchone()[0]==1

def test_stream_disconnect_and_fallback_costs_are_not_freed(harness):
    client,server,upstream,config=harness
    server.gateway_auth.clients[0][2].update(max_cost_usd_24h=10)
    def truncated(handler,body):
        handler.send_response(200);handler.send_header('Content-Type','text/event-stream');handler.end_headers()
        handler.wfile.write(b'data: {"type":"message_start","message":{"usage":{"input_tokens":10,"output_tokens":0}}}\n\n')
    upstream.respond=truncated
    with client.stream('POST','/v1/messages',json={**payload(),'stream':True}) as response:
        list(response.iter_bytes())
    budget=client.get('/v1/nexus/status').json()['budget']
    assert budget['reservations'][0]['state']=='uncertain'
    assert budget['held_microusd']>0

def test_legacy_usage_and_server_tools_fail_closed(harness):
    client,server,upstream,config=harness
    assert client.post('/v1/messages',json=payload()).status_code==200
    server.gateway_auth.clients[0][2].update(max_cost_usd_24h=10)
    assert client.post('/v1/messages',json=payload()).status_code==402
    body=payload();body['tools']=[{'type':'web_search_20250305','name':'web_search'}]
    assert client.post('/v1/messages',json=body).status_code==400
    assert len(upstream.requests)==1

def test_provider_overrun_stops_further_admission(harness):
    client,server,upstream,config=harness
    item=classification(server)
    ledger=BudgetLedger();ledger.reserve(item,{'max_cost_usd_24h':100},config,payload())
    ledger.attempt(item.request_id);ledger.finish(item.request_id,cost=10,complete=True)
    assert ledger.status(item.principal)['reservations'][0]['state']=='breach'
    with pytest.raises(PolicyError,match='exceeded'):
        ledger.reserve(classification(server),{'max_cost_usd_24h':100},config,payload())

from test_gateway_clients import native

def test_native_responses_budget_accounts_cached_tokens_once(native):
    client,server,upstream=native
    for model in server.gateway.config['models'].values():
        model['pricing']={'input':3,'output':15,'cache_read':.3,'cache_write_5m':3.75,'cache_write_1h':6}
    server.gateway_auth.clients[0][2]['max_cost_usd_24h']=10
    response=client.post('/v1/responses',json={'model':'nexus-auto','input':'Summarize this report','max_output_tokens':4096})
    assert response.status_code==200,response.text
    budget=client.get('/v1/nexus/status').json()['budget']
    assert budget['held_microusd']==159
    assert budget['reservations'][0]['state']=='settled'

def test_success_after_fallback_keeps_unknown_attempt_cost(harness):
    client,server,upstream,config=harness
    server.gateway_auth.clients[0][2]['max_cost_usd_24h']=10
    original=upstream.respond
    def missing(handler,body):
        if 'haiku' in handler.path:
            handler.send_response(404);handler.end_headers()
        else:original(handler,body)
    upstream.respond=missing
    assert client.post('/v1/messages',json=payload('Rewrite this email')).status_code==200
    row=client.get('/v1/nexus/status').json()['budget']['reservations'][0]
    assert row['state']=='uncertain' and row['attempts']==2
    assert row['held_microusd']==row['reserved_microusd']

def test_parallel_first_reservations_cannot_overspend(harness):
    client,server,upstream,config=harness
    def reserve(_):
        item=classification(server)
        try:BudgetLedger().reserve(item,{'max_cost_usd_24h':3},config,payload())
        except PolicyError:return None
        return item.request_id
    with ThreadPoolExecutor(max_workers=8) as pool:
        accepted=[item for item in pool.map(reserve,range(8)) if item]
    assert len(accepted)==1

def test_real_crashed_process_reservation_survives_and_reconciles(harness):
    import json,subprocess,sys
    client,server,upstream,config=harness
    code='''import json,os,sys
from pathlib import Path
from src import db
from src.gateway_policy import GatewayPolicy
from src.gateway_budget import BudgetLedger
args=json.load(sys.stdin)
db.DB_PATH=Path(args['path'])
item=GatewayPolicy(args['config']).classify({'model':'nexus-auto','messages':[{'role':'user','content':'Summarize'}]}, {}, 'crashed-owner')
ledger=BudgetLedger()
ledger.reserve(item,{'max_cost_usd_24h':10},args['config'],{'max_tokens':4096})
ledger.attempt(item.request_id)
print(item.request_id,flush=True)
os._exit(0)
'''
    result=subprocess.run([sys.executable,'-c',code],input=json.dumps({'path':str(db.DB_PATH),'config':config}),
                          text=True,capture_output=True,timeout=15)
    assert result.returncode==0,result.stderr
    request_id=result.stdout.strip()
    ledger=BudgetLedger()
    row=ledger.status('crashed-owner')['reservations'][0]
    assert row['request_id']==request_id and row['state']=='reserved'
    ledger.reconcile(request_id,.2,'Provider usage verified after process crash')
    assert ledger.status('crashed-owner')['held_microusd']==200000

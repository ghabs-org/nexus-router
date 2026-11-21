import copy
import json
import threading
from concurrent.futures import ThreadPoolExecutor
import pytest
from test_gateway import harness,payload
from src import db,gateway
from src.gateway_economics import fingerprint


def test_economic_order_requires_evaluated_quality_and_retains_pins(harness):
    client,server,upstream,config=harness
    config['economics']={'enabled':True,'minimum_saving_fraction':.1}
    server.gateway=gateway.GatewayProxy(config)
    preview=client.post('/v1/nexus/route-preview',json=payload()).json()
    assert not preview['explanation']['economics']['applied']
    for key,model in config['models'].items():
        model['quality_pass_rate']=.95
        if 'opus' in key:
            model['pricing']={k:v/10 for k,v in model['pricing'].items()}
    server.gateway=gateway.GatewayProxy(config)
    preview=client.post('/v1/nexus/route-preview',json=payload()).json()
    assert preview['explanation']['economics']['applied']
    assert 'opus' in preview['choices'][0]['target']
    response=client.post('/v1/messages',json=payload(),headers={'x-session-id':'economic-task'})
    assert response.status_code==200
    config['economics']['enabled']=False
    server.gateway=gateway.GatewayProxy(config)
    response=client.post('/v1/messages',json=payload(),headers={'x-session-id':'economic-task'})
    assert 'opus' in response.headers['x-nexus-routed-model']


def test_cache_hint_identity_includes_reasoning_and_principal():
    body=payload();choice={'target':'vertex:sonnet','effort':'medium'}
    first,_=fingerprint(body,choice,'alice','messages')
    assert first!=fingerprint(body,choice,'bob','messages')[0]
    assert first!=fingerprint(body,{**choice,'effort':'high'},'alice','messages')[0]
    assert first!=fingerprint(body,choice,'alice','responses')[0]


def ticket(client,session):
    response=client.post('/v1/nexus/escalations',json={'source_session':session,'wire':'messages'})
    assert response.status_code==200,response.text
    return response.json()


def test_escalation_is_bounded_fresh_and_one_shot(harness):
    client,server,upstream,config=harness
    assert client.post('/v1/messages',json=payload('Rewrite this email')).status_code==200
    session=client.get('/v1/nexus/status').json()['recent_turns'][0]['session']
    first=ticket(client,session)
    headers={'x-session-id':first['new_session_id'],'x-nexus-escalation-ticket':first['ticket']}
    bad=payload();bad['messages'].append({'role':'assistant','content':'old action'})
    assert client.post('/v1/messages',json=bad,headers=headers).status_code==409
    response=client.post('/v1/messages',json=payload(),headers=headers)
    assert response.status_code==200 and response.headers['x-nexus-profile']=='standard'
    assert client.post('/v1/messages',json=payload(),headers=headers).status_code==409
    session=client.get('/v1/nexus/status').json()['recent_turns'][0]['session']
    second=ticket(client,session)
    response=client.post('/v1/messages',json=payload(),headers={'x-session-id':second['new_session_id'],'x-nexus-escalation-ticket':second['ticket']})
    assert response.status_code==200 and response.headers['x-nexus-profile']=='deep'
    session=client.get('/v1/nexus/status').json()['recent_turns'][0]['session']
    assert client.post('/v1/nexus/escalations',json={'source_session':session}).status_code==409
    assert len(upstream.requests)==3


def test_escalation_tenant_isolation_and_protocol(harness):
    client,server,upstream,config=harness
    client.post('/v1/messages',json=payload())
    session=client.get('/v1/nexus/status').json()['recent_turns'][0]['session']
    assert client.post('/v1/nexus/escalations',json={'source_session':session},headers={'Authorization':'Bearer other-secret'}).status_code==404
    assert client.post('/v1/nexus/escalations',json={'source_session':session,'wire':'responses'}).status_code==400
    issued=ticket(client,session)
    response=client.post('/v1/messages',json=payload(),headers={'Authorization':'Bearer other-secret','x-session-id':issued['new_session_id'],'x-nexus-escalation-ticket':issued['ticket']})
    assert response.status_code==404


def test_dashboard_serves_no_secrets_and_uses_safe_dom(harness):
    client,server,upstream,config=harness
    page=client.get('/dashboard',headers={'Authorization':''})
    assert page.status_code==200 and 'frame-ancestors' in page.headers['content-security-policy']
    script=client.get('/dashboard.js').text
    assert 'innerHTML' not in script and 'localStorage' not in script and 'sessionStorage' not in script
    assert 'client-secret' not in page.text+script
    assert client.get('/v1/nexus/status',headers={'Authorization':''}).status_code==401

def test_feedback_is_private_and_never_runs_inference(harness):
    client,server,upstream,config=harness
    client.post('/v1/messages',json=payload())
    session=client.get('/v1/nexus/status').json()['recent_turns'][0]['session']
    response=client.post('/v1/nexus/feedback',json={'source_session':session,'rating':'poor_quality'})
    assert response.status_code==200
    assert len(upstream.requests)==1
    assert client.get('/v1/nexus/status').json()['feedback'][0]['rating']=='poor_quality'
    assert client.get('/v1/nexus/status',headers={'Authorization':'Bearer other-secret'}).json()['feedback']==[]


def test_mcp_escalation_starts_fresh_task(harness):
    from src.gateway_mcp import dispatch
    client,server,upstream,config=harness
    client.post('/v1/messages',json=payload('Rewrite this email'))
    session=client.get('/v1/nexus/status').json()['recent_turns'][0]['session']
    response=dispatch({'method':'tools/call','params':{'name':'nexus_escalate_task','arguments':{'source_session':session,'task':'Review this draft carefully'}}},client)
    assert not response.get('isError')
    assert len(upstream.requests)==2
    assert upstream.requests[-1][2]['messages']==[{'role':'user','content':'Review this draft carefully'}]

def test_warm_prefix_cost_hint_expires(harness):
    import time
    client,server,upstream,config=harness
    config['economics']={'enabled':True,'expected_output_tokens':1,'minimum_saving_fraction':.1}
    for key,model in config['models'].items():
        model['quality_pass_rate']=1
        if 'opus' in key:model['pricing']={k:v*1.1 for k,v in model['pricing'].items()}
    server.gateway=gateway.GatewayProxy(config)
    body=payload();body['system'][0]['text']='Common company instructions. '*1000
    preview=client.post('/v1/nexus/route-preview',json=body).json()
    assert 'sonnet' in preview['choices'][0]['target']
    principal,_=server.gateway_auth.authenticate({'Authorization':'Bearer client-secret'})
    choice=config['profiles']['standard'][1]
    key,prefix=fingerprint(body,choice,principal,'messages',config['providers']['vertex'])
    with db._connect() as conn:
        conn.execute('INSERT INTO gateway_cache_hints VALUES (?,?,?)',(key,prefix,time.time()))
    preview=client.post('/v1/nexus/route-preview',json=body).json()
    assert 'opus' in preview['choices'][0]['target']
    with db._connect() as conn:conn.execute('UPDATE gateway_cache_hints SET observed_at=?',(time.time()-1000,))
    preview=client.post('/v1/nexus/route-preview',json=body).json()
    assert 'sonnet' in preview['choices'][0]['target']


def test_ticket_concurrent_consumption_is_single_use(harness):
    client,server,upstream,config=harness
    client.post('/v1/messages',json=payload('Rewrite this email'))
    source=client.get('/v1/nexus/status').json()['recent_turns'][0]['session']
    issued=ticket(client,source)
    headers={'x-session-id':issued['new_session_id'],'x-nexus-escalation-ticket':issued['ticket']}
    with ThreadPoolExecutor(max_workers=2) as workers:
        responses=list(workers.map(lambda _:client.post('/v1/messages',json=payload(),headers=headers).status_code,range(2)))
    assert sorted(responses)==[200,409]
    assert len(upstream.requests)==2

def test_escalation_keeps_original_task_spending_allowance(harness):
    client,server,upstream,config=harness
    server.gateway_auth.clients[0][2]['max_task_cost_usd']=2.5229
    assert client.post('/v1/messages',json=payload('Rewrite this email')).status_code==200
    session=client.get('/v1/nexus/status').json()['recent_turns'][0]['session']
    issued=ticket(client,session)
    response=client.post('/v1/messages',json=payload(),headers={'x-session-id':issued['new_session_id'],'x-nexus-escalation-ticket':issued['ticket']})
    assert response.status_code==402,response.text
    assert len(upstream.requests)==1

def test_economics_does_not_trade_quality_for_price(harness):
    client,server,upstream,config=harness
    config['economics']={'enabled':True}
    for key,model in config['models'].items():
        model['quality_pass_rate']=.95
        if 'opus' in key:
            model['quality_pass_rate']=.5
            model['pricing']={k:v/100 for k,v in model['pricing'].items()}
    server.gateway=gateway.GatewayProxy(config)
    preview=client.post('/v1/nexus/route-preview',json=payload()).json()
    assert 'sonnet' in preview['choices'][0]['target']
    assert not preview['explanation']['economics']['applied']

def test_all_economic_candidates_below_floor_fail_closed(harness):
    client,server,upstream,config=harness
    config['economics']={'enabled':True}
    for model in config['models'].values():model['quality_pass_rate']=.5
    server.gateway=gateway.GatewayProxy(config)
    assert client.post('/v1/messages',json=payload()).status_code==503
    assert not upstream.requests

def test_dashboard_logout_ignores_inflight_status():
    import shutil,subprocess
    from pathlib import Path
    node=shutil.which('node')
    if not node:pytest.skip('Node required for dashboard behavior check')
    source=Path(__file__).parents[1]/'src/gateway_dashboard.js'
    code=r'''
const fs=require('fs'),vm=require('vm'),assert=require('assert');
const nodes={};
function element(){return {textContent:'',value:'',children:[],append(x){this.children.push(x)},replaceChildren(){this.children=[]}}}
const document={getElementById(id){return nodes[id]??=element()},createElement:element};
let complete;
const context=vm.createContext({document,fetch:()=>new Promise(resolve=>{complete=resolve})});
vm.runInContext(fs.readFileSync(process.argv[1],'utf8'),context);
document.getElementById('token').value='test-only';
nodes.login.onsubmit({preventDefault(){}});
nodes.logout.onclick();
complete({ok:true,json:async()=>({usage_24h:{private:'old account'},budget:{held_microusd:0,reservations:[]},recent_turns:[]})});
setImmediate(()=>{assert.strictEqual(nodes.summary.textContent,'');assert.strictEqual(nodes.routes.children.length,0)});
'''
    result=subprocess.run([node,'-e',code,str(source)],text=True,capture_output=True,timeout=10)
    assert result.returncode==0,result.stderr

def test_escalated_task_followups_keep_pin_without_reclaiming_ticket(harness):
    client,server,upstream,config=harness
    client.post('/v1/messages',json=payload('Rewrite this email'))
    session=client.get('/v1/nexus/status').json()['recent_turns'][0]['session']
    issued=ticket(client,session)
    assert client.post('/v1/messages',json=payload(),headers={'x-session-id':issued['new_session_id'],'x-nexus-escalation-ticket':issued['ticket']}).status_code==200
    response=client.post('/v1/messages',json=payload('Continue the task'),headers={'x-session-id':issued['new_session_id']})
    assert response.status_code==200,response.text
    assert response.headers['x-nexus-profile']=='standard'


def test_effort_feedback_ratings_summary_and_cli(harness,tmp_path,capsys):
    from src.gateway_cli import main as cli_main
    client,server,upstream,config=harness
    assert client.post('/v1/messages',json=payload('Summarize this note')).status_code==200
    session=client.get('/v1/nexus/status').json()['recent_turns'][0]['session']
    assert client.post('/v1/nexus/feedback',json={'source_session':session,'rating':'meh'}).status_code==400
    for rating in ('underthinking','underthinking','underthinking','overthinking'):
        assert client.post('/v1/nexus/feedback',json={'source_session':session,'rating':rating}).status_code==200
    status=client.get('/v1/nexus/status').json()
    summary=status['feedback_summary']
    assert any(r['profile']=='standard' and r['effort']=='medium' and r['rating']=='underthinking' and r['count']==3 for r in summary)
    assert client.get('/v1/nexus/status',headers={'Authorization':'Bearer other-secret'}).json()['feedback_summary']==[]

    db_dir=tmp_path/'cli-state'
    (db_dir/'data').mkdir(parents=True)
    import sqlite3
    with sqlite3.connect(db.DB_PATH) as src, sqlite3.connect(db_dir/'data/router.sqlite') as dst:
        src.backup(dst)
    capsys.readouterr()
    assert cli_main(['feedback-summary','--state-dir',str(db_dir)])==0
    cli_out=json.loads(capsys.readouterr().out)
    assert cli_out['by_rating']['underthinking']==3
    assert cli_out['recommendations'][0]['signal']=='underthinking'


def test_dashboard_effort_feedback_buttons_and_safe_dom():
    import shutil,subprocess
    from pathlib import Path
    node=shutil.which('node')
    if not node:pytest.skip('Node required for dashboard behavior check')
    source=Path(__file__).parents[1]/'src/gateway_dashboard.js'
    script=source.read_text()
    assert 'innerHTML' not in script and 'localStorage' not in script and 'sessionStorage' not in script
    code=r'''
const fs=require('fs'),vm=require('vm'),assert=require('assert');
const nodes={},requests=[];
function element(){return {textContent:'',value:'',children:[],append(x){this.children.push(x)},replaceChildren(){this.children=[]}}}
const document={getElementById(id){return nodes[id]??=element()},createElement:element};
const context=vm.createContext({document,fetch:(path,opts)=>{requests.push({path,body:opts?.body?JSON.parse(opts.body):null});return Promise.resolve({ok:true,json:async()=>({usage_24h:{},budget:{held_microusd:0,reservations:[]},recent_turns:[{session:'s1',profile:'standard',target:'vertex:claude-sonnet-4-6',effort:'medium',http_status:200,explanation:{source:'default'}}],feedback_summary:[{profile:'standard',effort:'medium',rating:'underthinking',count:1}]})})}});
vm.runInContext(fs.readFileSync(process.argv[1],'utf8'),context);
document.getElementById('token').value='test-only';
nodes.login.onsubmit({preventDefault(){}});
setImmediate(async()=>{
  const row=nodes.routes.children[0];
  const actions=row.children[3].children.map(b=>b.textContent);
  assert.deepStrictEqual(actions,['Good','Underthought','Overthought','Too expensive','Poor quality']);
  await row.children[3].children[1].onclick();
  assert.deepStrictEqual(requests.at(-1),{path:'/v1/nexus/feedback',body:{source_session:'s1',rating:'underthinking'}});
});
'''
    result=subprocess.run([node,'-e',code,str(source)],text=True,capture_output=True,timeout=10)
    assert result.returncode==0,result.stderr

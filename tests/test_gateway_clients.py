import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import tomllib

import pytest
from test_gateway import harness, payload
from src import db, gateway
from src.gateway_cli import main
from src.gateway_mcp import dispatch

@pytest.fixture
def native(harness, monkeypatch):
    client, server, upstream, config = harness
    sample = gateway.load_config(Path(__file__).parents[1] / 'src/gateway.codex.yaml')
    monkeypatch.setenv('OPENAI_API_KEY', 'upstream-openai-test')
    sample['providers']['openai']['base_url'] = f'http://127.0.0.1:{upstream.server_port}'
    config['providers'].update(sample['providers'])
    config['models'].update(sample['models'])
    config['protocol_profiles'] = sample['protocol_profiles']
    server.gateway = gateway.GatewayProxy(config)
    db.upsert_provider_health_state('openai', {'auth':'ok','quota':'healthy','health_score':1})
    with db._connect() as conn:
        conn.execute("UPDATE providers SET enabled=1,status='enabled' WHERE provider='openai'")
    def answer(handler, body):
        handler.send_response(200)
        streaming = body.get('stream')
        handler.send_header('Content-Type', 'text/event-stream' if streaming else 'application/json')
        handler.end_headers()
        data = {'id':'resp-1','output':[], 'usage':{'input_tokens':100,'output_tokens':5,'input_tokens_details':{'cached_tokens':80}}}
        if streaming:
            data = {'type':'response.completed','response':data}
            handler.wfile.write(('data: '+json.dumps(data)+'\n\n').encode())
        else:
            handler.wfile.write(json.dumps(data).encode())
    upstream.respond = answer
    return client, server, upstream

@pytest.mark.parametrize('path', ['/v1/responses','/v1/chat/completions'])
def test_native_model_effort_cache_and_auth(native, path):
    client, server, upstream = native
    body = {'model':'nexus-auto','input':'Analyze architecture','max_output_tokens':4096,'prompt_cache_key':'stable-key'} if path.endswith('responses') else {'model':'nexus-auto','messages':[{'role':'user','content':'Analyze architecture'}],'max_completion_tokens':4096}
    response = client.post(path,json=body,headers={'x-session-id':'codex-task'})
    assert response.status_code == 200, response.text
    endpoint, headers, forwarded = upstream.requests[-1]
    assert endpoint == path
    assert headers['authorization'] == 'Bearer upstream-openai-test'
    assert forwarded['model'] == 'gpt-6.1-sol'
    assert 'thinking' not in forwarded and 'output_config' not in forwarded
    if path.endswith('responses'):
        assert forwarded['reasoning']['effort'] == 'high'
        assert forwarded['prompt_cache_key'] == 'stable-key'
    else:
        assert forwarded['reasoning_effort'] == 'high'
    status = client.get('/v1/nexus/status').json()
    assert status['usage_24h']['cache_read_input_tokens'] == 80
    assert status['usage_24h']['input_tokens'] == 100
    assert client.post(path,json=body,headers={'x-session-id':'codex-task'}).headers['x-nexus-cache-pinned'] == 'true'

def test_responses_continuations_and_streaming(native):
    client, server, upstream = native
    body = {'model':'nexus-auto','input':'Analyze architecture','stream':True}
    with client.stream('POST','/v1/responses',json=body,headers={'x-session-id':'chain'}) as response:
        assert response.status_code == 200
        assert b'response.completed' in b''.join(response.iter_bytes())
    follow = {'model':'nexus-auto','input':[{'type':'function_call_output','call_id':'call-1','output':'ok'}], 'previous_response_id':'resp-1'}
    assert client.post('/v1/responses',json=follow,headers={'x-session-id':'chain'}).status_code == 200
    assert upstream.requests[-1][2]['previous_response_id'] == 'resp-1'
    assert upstream.requests[-1][2]['input'] == follow['input']
    assert client.post('/v1/responses',json=follow,headers={'x-session-id':'unknown'}).status_code == 409

def test_antigravity_task_tool_calls_real_gateway(harness):
    client, server, upstream, config = harness
    result = dispatch({'method':'tools/call','params':{'name':'nexus_run_task','arguments':{'task':'Rewrite this email'}}},client)
    assert not result.get('isError')
    assert 'haiku' in upstream.requests[-1][0]
    assert upstream.requests[-1][2]['tools'] == []
    assert dispatch({'method':'tools/call','params':{'name':'nexus_status','arguments':{}}},client)['content']

def test_client_config_and_stdio(tmp_path,monkeypatch):
    monkeypatch.setenv('NEXUS_ROUTER_STATE_ROOT',str(tmp_path))
    monkeypatch.setenv('NEXUS_ROUTER_DB_PATH',str(tmp_path/'db'))
    for command, suffix in [('codex-config','toml'),('antigravity-config','json')]:
        output=tmp_path/f'{command}.{suffix}'
        assert main([command,'--output',str(output)]) == 0
        if suffix=='toml':
            config=tomllib.loads(output.read_text())
            assert config['model_providers']['nexus']['wire_api']=='responses'
            assert config['model_providers']['nexus']['env_key']=='NEXUS_GATEWAY_TOKEN'
        else:
            assert json.loads(output.read_text())['mcpServers']['nexus-router']['args'][0]=='mcp'
    env={**os.environ,'NEXUS_GATEWAY_TOKEN':'test-token'}
    process=subprocess.run([sys.executable,'-m','src.gateway_cli','mcp'],input='{"jsonrpc":"2.0","id":1,"method":"initialize","params":{}}\n{"jsonrpc":"2.0","id":2,"method":"tools/list"}\n',text=True,capture_output=True,env=env)
    assert process.returncode==0,process.stderr
    replies=[json.loads(line) for line in process.stdout.splitlines()]
    assert replies[0]['result']['capabilities']['tools']=={}
    assert replies[1]['result']['tools'][0]['name']=='nexus_run_task'


def test_stdio_task_call_through_tcp_gateway(harness):
    if os.environ.get('NEXUS_TEST_SOCKETPAIR') == '1':
        pytest.skip('Subprocess attachment requires TCP; socket-pair registries are process-local')
    client, server, upstream, config = harness
    requests = [
        {'jsonrpc':'2.0','id':1,'method':'initialize','params':{}},
        {'jsonrpc':'2.0','id':2,'method':'tools/call','params':{
            'name':'nexus_run_task','arguments':{'task':'Rewrite this email','session_id':'stdio-smoke'}}},
    ]
    env = {**os.environ, 'NEXUS_GATEWAY_TOKEN':'client-secret'}
    result = subprocess.run([sys.executable,'-m','src.gateway_cli','mcp','--url',str(client.base_url).rstrip('/')],
                            input=''.join(json.dumps(item)+'\n' for item in requests), text=True,
                            capture_output=True, env=env, timeout=15)
    assert result.returncode == 0, result.stderr
    replies = [json.loads(line) for line in result.stdout.splitlines()]
    assert 'error' not in replies[1], replies[1]
    assert not replies[1]['result'].get('isError'), replies[1]
    answer = json.loads(replies[1]['result']['content'][0]['text'])
    assert answer['content'][0]['text'] == 'Done'
    assert len(upstream.requests) == 1 and 'haiku' in upstream.requests[0][0]
    status = client.get('/v1/nexus/status').json()
    assert status['usage_24h']['cache_read_input_tokens'] == 30

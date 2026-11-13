#!/usr/bin/env python3
"""Exercise the installed Codex CLI against the real gateway and a local SSE fixture.

No external inference call is made. No user client settings or credentials change.
"""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src import db, gateway

class Upstream(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def do_POST(self):
        request = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
        self.server.seen.append({'path': self.path, 'model': request.get('model'), 'effort': request.get('reasoning', {}).get('effort')})
        part = {'type':'output_text', 'text':'NEXUS_CLIENT_SMOKE_OK', 'annotations':[], 'logprobs':[]}
        item = {'id':'msg_smoke', 'type':'message', 'role':'assistant', 'status':'completed', 'content':[part]}
        response = {'id':'resp_smoke', 'object':'response', 'created_at':1, 'status':'completed', 'model':request['model'],
                    'output':[item], 'usage':{'input_tokens':100, 'output_tokens':5, 'total_tokens':105, 'input_tokens_details':{'cached_tokens':40}}}
        events = [
            {'type':'response.created', 'response':{**response,'status':'in_progress','output':[]}},
            {'type':'response.output_item.added','output_index':0,'item':{**item,'status':'in_progress','content':[]}},
            {'type':'response.content_part.added','item_id':'msg_smoke','output_index':0,'content_index':0,'part':{**part,'text':''}},
            {'type':'response.output_text.delta','item_id':'msg_smoke','output_index':0,'content_index':0,'delta':part['text']},
            {'type':'response.output_text.done','item_id':'msg_smoke','output_index':0,'content_index':0,'text':part['text']},
            {'type':'response.content_part.done','item_id':'msg_smoke','output_index':0,'content_index':0,'part':part},
            {'type':'response.output_item.done','output_index':0,'item':item},
            {'type':'response.completed','response':response},
        ]
        self.send_response(200)
        self.send_header('Content-Type','text/event-stream')
        self.end_headers()
        for index, event in enumerate(events):
            event['sequence_number'] = index
            self.wfile.write(('event: '+event['type']+'\ndata: '+json.dumps(event)+'\n\n').encode())
            self.wfile.flush()

def main():
    executable = shutil.which('codex')
    if not executable:
        raise SystemExit('Codex CLI is not installed')
    with tempfile.TemporaryDirectory(prefix='nexus-codex-smoke-') as directory:
        previous_db = db.DB_PATH
        previous_env = {name: os.environ.get(name) for name in ('NEXUS_GATEWAY_TOKEN','OPENAI_API_KEY')}
        upstream = server = None
        try:
            db.DB_PATH = Path(directory)/'router.sqlite'
            os.environ.update(NEXUS_GATEWAY_TOKEN='nexus-smoke-client', OPENAI_API_KEY='nexus-smoke-upstream')
            config = gateway.load_config(ROOT/'src/gateway.codex.yaml')
            upstream = ThreadingHTTPServer(('127.0.0.1',0),Upstream)
            upstream.seen=[]
            config['providers']['openai']['base_url']=f'http://127.0.0.1:{upstream.server_port}'
            server=gateway.create_server(config,port=0)
            db.upsert_provider_health_state('openai',{'auth':'ok','quota':'healthy','health_score':1})
            with db._connect() as conn:
                conn.execute("UPDATE providers SET enabled=1,status='enabled' WHERE provider='openai'")
            for instance in (upstream,server):
                threading.Thread(target=instance.serve_forever,daemon=True).start()
            args=[executable,'exec','--ignore-user-config','--ignore-rules','--ephemeral','--skip-git-repo-check',
                  '--sandbox','read-only','-C',directory,'-c','model="nexus-auto"','-c','model_provider="nexus"',
                  '-c','model_providers.nexus.name="Nexus smoke"','-c','model_providers.nexus.wire_api="responses"',
                  '-c',f'model_providers.nexus.base_url="http://127.0.0.1:{server.server_port}/v1"',
                  '-c','model_providers.nexus.env_key="NEXUS_GATEWAY_TOKEN"',
                  '-c','model_providers.nexus.requires_openai_auth=false',
                  'Reply with NEXUS_CLIENT_SMOKE_OK only. Do not call any tools.']
            result=subprocess.run(args,env=dict(os.environ),capture_output=True,text=True,stdin=subprocess.DEVNULL,timeout=45)
            if result.returncode or 'NEXUS_CLIENT_SMOKE_OK' not in result.stdout or not upstream.seen:
                print(json.dumps({'passed':False,'exit_code':result.returncode,'upstream_requests':len(upstream.seen),
                                  'diagnostic':result.stderr[-2000:]}))
                return 1
            print(json.dumps({'passed':True,'mode':'installed Codex CLI + real TCP gateway + local upstream fixture',
                              'routes':upstream.seen}))
            return 0
        finally:
            for instance in (server,upstream):
                if instance:
                    instance.shutdown(); instance.server_close()
            db.DB_PATH=previous_db
            for key,value in previous_env.items():
                if value is None: os.environ.pop(key,None)
                else: os.environ[key]=value

if __name__=='__main__':
    raise SystemExit(main())

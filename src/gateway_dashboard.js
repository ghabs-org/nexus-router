'use strict';
let token='';
let generation=0;
const el=id=>document.getElementById(id);
function clearView(){el('routes').replaceChildren();for(const id of ['summary','budget','ticket','message'])el(id).textContent='';}
async function request(path,body){
  if(!token)throw new Error('Enter a gateway token.');
  const response=await fetch(path,{method:body?'POST':'GET',credentials:'omit',cache:'no-store',headers:{Authorization:'Bearer '+token,...(body?{'Content-Type':'application/json'}:{})},body:body?JSON.stringify(body):undefined});
  const data=await response.json();
  if(!response.ok)throw new Error(data.error?.message||'Request failed: '+response.status);
  return data;
}
async function refresh(){
  const started=generation;
  try{
    const data=await request('/v1/nexus/status');
    if(started!==generation)return;
    el('message').textContent='Loaded your gateway status.';
    el('summary').textContent=JSON.stringify({usage_24h:data.usage_24h,held_microusd:data.budget.held_microusd},null,2);
    el('budget').textContent=JSON.stringify(data.budget.reservations,null,2);
    el('routes').replaceChildren();
    for(const row of data.recent_turns){
      const tr=document.createElement('tr');
      for(const value of [row.profile+' / '+(row.target||'unavailable'),(row.effort||'default')+' / HTTP '+row.http_status,JSON.stringify(row.explanation)]){
        const td=document.createElement('td');td.textContent=value;tr.append(td);
      }
      const td=document.createElement('td');
      const feedback=document.createElement('button');feedback.textContent='Report poor result';
      feedback.onclick=async()=>{const current=generation;try{await request('/v1/nexus/feedback',{source_session:row.session,rating:'poor_quality'});if(current!==generation)return;el('message').textContent='Feedback recorded; no work was replayed.';}catch(error){if(current===generation)el('message').textContent=error.message;}};
      td.append(feedback);
      if(['utility','standard'].includes(row.profile)&&row.explanation?.wire){
        const button=document.createElement('button');button.textContent='Request stronger route';
        button.onclick=async()=>{const current=generation;try{const ticket=await request('/v1/nexus/escalations',{source_session:row.session,wire:row.explanation.wire});if(current!==generation)return;el('ticket').textContent=JSON.stringify(ticket,null,2);}catch(error){if(current===generation)el('message').textContent=error.message;}};
        td.append(button);
      }
      tr.append(td);el('routes').append(tr);
    }
  }catch(error){if(started===generation)el('message').textContent=error.message;}
}
el('login').onsubmit=event=>{event.preventDefault();generation++;token=el('token').value;el('token').value='';clearView();refresh();};
el('refresh').onclick=refresh;
el('logout').onclick=()=>{generation++;token='';clearView();};

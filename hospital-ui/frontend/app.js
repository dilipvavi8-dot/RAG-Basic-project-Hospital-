'use strict';
const $ = (id) => document.getElementById(id);
let conversationId = null;
let busy = false;
let pending = null;

function node(tag, text, className) {
  const element = document.createElement(tag);
  if (text !== undefined) element.textContent = text;
  if (className) element.className = className;
  return element;
}
function setBusy(value) {
  busy = value;
  $('send').disabled = value;
  $('new-chat').disabled = value;
  $('retry').disabled = value;
  document.querySelectorAll('[data-question]').forEach(button => button.disabled = value);
  $('chat-form').setAttribute('aria-busy', String(value));
}
async function api(path, body) {
  let response;
  try {
    response = await fetch(path, {
      method: body === undefined ? 'GET' : 'POST',
      headers: {'Content-Type': 'application/json'},
      body: body === undefined ? undefined : JSON.stringify(body),
      signal: AbortSignal.timeout(610000),
    });
  } catch {
    throw new Error('Cannot reach the local backend. Check that the server is running, then retry.');
  }
  const data = await response.json();
  if (!response.ok) throw new Error(typeof data.detail === 'string' ? data.detail : 'Request failed. Please try again.');
  return data;
}
function error(message) {
  $('error-text').textContent = message;
  $('error').hidden = false;
  $('retry').hidden = !pending;
}
function bubble(role, text) {
  $('welcome').hidden = true;
  const item = node('article', undefined, `message ${role}`);
  item.append(node('div', role === 'user' ? 'You' : 'Hospital assistant', 'speaker'));
  item.append(node('div', text, 'text'));
  $('messages').append(item);
  item.scrollIntoView({block: 'nearest', behavior: 'smooth'});
  return item;
}
function showAnswer(data) {
  const item = bubble('assistant', data.answer);
  if (data.sources.length) {
    const references = node('details');
    const cited = data.sources.filter(source => source.cited).length;
    references.append(node('summary', `References · ${cited} cited / ${data.sources.length} retrieved`));
    references.append(node('p', 'Numbers apply only to this answer. “Cited” means its label appears in the answer; it does not prove the claim is correct.', 'citation-note'));
    data.sources.forEach(source => {
      const row = node('div', undefined, 'source');
      row.append(node('strong', `[${source.label}] ${source.filename} · ${source.cited ? 'Cited in answer' : 'Retrieved only'}`));
      row.append(node('div', [source.section, source.subsection].filter(Boolean).join(' / ') || 'No section metadata'));
      references.append(row);
    });
    item.append(references);
  }
  const details = node('details');
  details.append(node('summary', 'Response details'));
  details.append(node('p', `Hospital search: ${data.search_used ? 'used' : 'not used'} · Response time: ${data.elapsed_seconds.toFixed(2)} seconds`));
  item.append(details);
}
async function send(retry = false) {
  if (busy) return;
  if (!retry) {
    const question = $('question').value.trim();
    if (!question) return;
    if (pending) { error('Retry the pending question or start a New Conversation before sending another.'); return; }
    pending = {question, request_id: crypto.randomUUID()};
    bubble('user', question);
    $('question').value = '';
  }
  if (!pending) return;
  setBusy(true);
  $('error').hidden = true;
  const loading = node('div', 'Waiting for the hospital assistant…', 'pending');
  loading.setAttribute('role', 'status');
  $('messages').append(loading);
  loading.scrollIntoView({block: 'nearest'});
  try {
    if (!conversationId) conversationId = (await api('/api/conversations', {})).conversation_id;
    const data = await api('/api/chat', {...pending, conversation_id: conversationId});
    showAnswer(data);
    pending = null;
  } catch (exc) { error(exc.message); }
  finally { loading.remove(); setBusy(false); $('question').focus(); }
}
async function newConversation() {
  if (busy) return;
  setBusy(true);
  try {
    conversationId = (await api('/api/conversations', {})).conversation_id;
    pending = null;
    $('messages').replaceChildren();
    $('welcome').hidden = false;
    $('error').hidden = true;
    $('question').value = '';
    $('connection').textContent = 'New conversation ready. Previous messages will not be used.';
    $('connection').classList.remove('problem');
  } catch (exc) { error(exc.message); }
  finally { setBusy(false); $('question').focus(); }
}
$('chat-form').addEventListener('submit', event => {event.preventDefault(); send();});
$('question').addEventListener('keydown', event => {
  if (event.key === 'Enter' && !event.shiftKey && !event.isComposing) {event.preventDefault(); send();}
});
$('new-chat').addEventListener('click', newConversation);
$('retry').addEventListener('click', () => send(true));
document.querySelectorAll('[data-question]').forEach(button => button.addEventListener('click', () => {
  $('question').value = button.dataset.question;
  $('question').focus();
}));
api('/api/health').then(data => {
  $('connection').textContent = data.message;
  $('connection').classList.toggle('problem', !data.ready);
}).catch(exc => {
  $('connection').textContent = exc.message;
  $('connection').classList.add('problem');
});

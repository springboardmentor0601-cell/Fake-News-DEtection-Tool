document.addEventListener('DOMContentLoaded', function(){
  const root = document.getElementById('tg-chat-root');
  const toggle = document.getElementById('tg-chat-toggle');
  const closeBtn = document.getElementById('tg-chat-close');
  const form = document.getElementById('tg-chat-form');
  const input = document.getElementById('tg-chat-input');
  const messages = document.getElementById('tg-chat-messages');

  function appendMessage(text, cls) {
    if (!messages) return;
    const el = document.createElement('div');
    el.className = 'tg-msg ' + (cls||'bot');
    el.textContent = text;
    messages.appendChild(el);
    messages.scrollTop = messages.scrollHeight;
  }

  if (toggle) toggle.addEventListener('click', ()=> root?.classList.add('open') );
  if (closeBtn) closeBtn.addEventListener('click', ()=> root?.classList.remove('open') );

  if (form) form.addEventListener('submit', async e => {
    e.preventDefault();
    const txt = (input.value||'').trim();
    if (!txt) return;
    appendMessage(txt, 'user');
    input.value = '';
    appendMessage('Typing...', 'bot');
    try {
      const resp = await fetch('/api/chat/send_public', {
        method:'POST',
        headers:{'Content-Type':'application/json'},
        body: JSON.stringify({message: txt})
      });
      const data = await resp.json();
      const bots = messages.querySelectorAll('.tg-msg.bot');
      if (bots.length) bots[bots.length-1].remove();
      appendMessage(data.success ? data.response : ('Error: '+(data.error||'no response')), 'bot');
    } catch (err) {
      appendMessage('Network error: ' + (err.message||err), 'bot');
    }
  });
});

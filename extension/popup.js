document.addEventListener('DOMContentLoaded', () => {
    const form = document.getElementById('chat-form');
    const input = document.getElementById('chat-input');
    const chatContainer = document.getElementById('chat-container');
    const loadingBarContainer = document.getElementById('loading-bar-container');
    const loadingBar = document.getElementById('loading-bar');
  
    form.addEventListener('submit', (e) => {
      e.preventDefault();
      const message = input.value.trim();
      if (!message) return;
  
      addMessage("You", message, "user");
      input.value = '';
  
      chrome.tabs.query({ active: true, currentWindow: true }, (tabs) => {
        if (tabs.length > 0) {
          const url = tabs[0].url;
          const coreDomain = extractCoreDomain(url);
          sendMessage(message, coreDomain);
        }
      });
    });
  
    function extractCoreDomain(url) {
      const match = url.match(/^(https?:\/\/[^/]+)/);
      return match ? match[1] : url;
    }
  
    function addMessage(sender, text, type) {
      const msgDiv = document.createElement('div');
      msgDiv.classList.add('message', type);
      msgDiv.textContent = text;
      chatContainer.appendChild(msgDiv);
      chatContainer.scrollTop = chatContainer.scrollHeight;
      return msgDiv;
    }
  
    function showLoadingBar() {
      loadingBarContainer.style.display = 'block';
      loadingBar.style.width = '0%';
      loadingBarContainer.classList.add('loading-active');
    }
  
    function hideLoadingBar() {
      loadingBarContainer.classList.remove('loading-active');
      loadingBarContainer.style.display = 'none';
      loadingBar.style.width = '0%';
    }
  
    // Create a typing indicator bubble for the bot
    function addTypingIndicator() {
      const typingDiv = document.createElement('div');
      typingDiv.classList.add('message', 'bot', 'typing');
      const indicator = document.createElement('div');
      indicator.classList.add('typing-indicator');
      indicator.innerHTML = '<span></span><span></span><span></span>';
      typingDiv.appendChild(indicator);
      chatContainer.appendChild(typingDiv);
      chatContainer.scrollTop = chatContainer.scrollHeight;
      return typingDiv;
    }
  
    function sendMessage(message, coreDomain) {
      showLoadingBar();
  
      // Add the typing indicator before sending the fetch request.
      const typingMsg = addTypingIndicator();
  
      fetch("https://google-extension-production.up.railway.app/generate", {
        method: "POST",
        headers: {
          "Content-Type": "application/json"
        },
        body: JSON.stringify({ message: message, url: coreDomain }),
        mode: "cors"
      })
      .then(response => response.json())
      .then(data => {
        hideLoadingBar();
        // Replace typing indicator with actual bot message
        typingMsg.textContent = data.answer;
        typingMsg.classList.remove("typing");
      })
      .catch(err => {
        hideLoadingBar();
        typingMsg.textContent = "An error occurred: " + err;
        typingMsg.classList.remove("typing");
      });
    }
  });
  
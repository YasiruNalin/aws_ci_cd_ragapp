document.addEventListener('DOMContentLoaded', () => {
    const sendBtn = document.getElementById('send-btn');
    const userInput = document.getElementById('user-input');
    const chatWindow = document.getElementById('chat-window');
    const threadsContainer = document.getElementById('threads');
    const newChatBtn = document.getElementById('new-chat-btn');
    const fileUpload = document.getElementById('file-upload');
    const promptsContainer = document.getElementById('prompts');
    const themeToggleBtn = document.getElementById('theme-toggle-btn');

    let currentThreadId = localStorage.getItem('currentThreadId') || createNewThread();
    let currentTheme = localStorage.getItem('theme') || 'light';
    document.body.classList.add(currentTheme);

    themeToggleBtn.addEventListener('click', toggleTheme);

    function createNewThread() {
        const threadId = `thread-${Date.now()}`;
        localStorage.setItem('currentThreadId', threadId);
        return threadId;
    }

    function loadThreads() {
        fetch('/conversations/')
            .then(response => response.json())
            .then(threads => {
                threadsContainer.innerHTML = '';
                threads.forEach(thread => {
                    const threadDiv = document.createElement('div');
                    threadDiv.classList.add('thread');
                    threadDiv.innerHTML = `Conversation ${thread.id} 
                    <button class="delete-btn" onclick="deleteThread(${thread.id})"><i class="fas fa-trash-alt"></i></button>`;
                    threadDiv.addEventListener('click', () => switchThread(thread.id));
                    threadsContainer.appendChild(threadDiv);
                });
            });
    }

    function switchThread(threadId) {
        currentThreadId = threadId;
        localStorage.setItem('currentThreadId', threadId);
        loadChatHistory(threadId);
    }

    function loadChatHistory(threadId) {
        fetch('/conversations/')
            .then(response => response.json())
            .then(conversations => {
                chatWindow.innerHTML = '';
                const chatHistory = conversations.find(convo => convo.id === threadId);
                if (chatHistory) {
                    addMessage(chatHistory.user_query, 'user');
                    addMessage(chatHistory.answer, 'bot');
                }
                promptsContainer.style.display = chatHistory ? 'none' : 'flex';
            });
    }

    window.deleteThread = function (threadId) {
        fetch(`/conversations/${threadId}`, {
            method: 'DELETE',
        }).then(() => {
            loadThreads();
            loadChatHistory(currentThreadId);
        });
    };

    function sendMessage() {
        const text = userInput.value.trim();
        if (text === '') return;

        addMessage(text, 'user');
        userInput.value = '';

        fetch('/query/', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ question: text })
        })
        .then(response => response.json())
        .then(data => {
            addMessage(data.answer, 'bot');
        });

        promptsContainer.style.display = 'none';
    }

    function addMessage(text, sender) {
        const message = document.createElement('div');
        message.classList.add('message', sender);
        message.innerText = text;
        chatWindow.appendChild(message);
        chatWindow.scrollTop = chatWindow.scrollHeight;
    }

    window.sendPrompt = function (prompt) {
        userInput.value = prompt;
        sendMessage();
    };

    fileUpload.addEventListener('change', (event) => {
        const file = event.target.files[0];
        const formData = new FormData();
        formData.append('file', file);

        fetch('/upload-pdf/', {
            method: 'POST',
            body: formData
        })
        .then(response => response.json())
        .then(data => {
            console.log('File uploaded:', data);
        });
    });

    function toggleTheme() {
        document.body.classList.toggle('light');
        document.body.classList.toggle('dark');
        currentTheme = document.body.classList.contains('light') ? 'light' : 'dark';
        localStorage.setItem('theme', currentTheme);
    }

    sendBtn.addEventListener('click', sendMessage);
    userInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter') sendMessage();
    });
    newChatBtn.addEventListener('click', () => {
        currentThreadId = createNewThread();
        loadThreads();
        chatWindow.innerHTML = '';
        promptsContainer.style.display = 'flex';
    });

    loadThreads();
    loadChatHistory(currentThreadId);
});

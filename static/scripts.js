document.addEventListener("DOMContentLoaded", () => {
    const tabs = document.querySelectorAll(".tab");
    const sections = document.querySelectorAll(".content-section");
    const chatMessages = document.getElementById("chat-messages");
    const chatInput = document.getElementById("chat-input");
    const sendButton = document.getElementById("send-button");
    const navigateButtons = document.querySelectorAll(".navigate-button");
    const agriculturalAreas = document.getElementById("agricultural-areas");
    const predictCropButton = document.getElementById('predict-crop-button');
    const predictPriceButton = document.getElementById('predict-price-button');
    const yearInput = document.getElementById('year');
    const monthInput = document.getElementById('month');
    const weekInput = document.getElementById('week');
    const cropInput = document.getElementById('crop'); // New crop input
    const sendBtn = document.getElementById('send-btn');
    const userInput = document.getElementById('user-input');
    const chatWindow = document.getElementById('chat-window');
    const fileUpload = document.getElementById('file-upload');
    const themeToggleBtn = document.getElementById('theme-toggle-btn');
    const cropRecommendations = document.getElementById('crop-recommendations');
    const pricePrediction = document.getElementById('price-prediction');

    let currentThreadId = localStorage.getItem('currentThreadId') || createNewThread();
    let currentTheme = localStorage.getItem('theme') || 'light';
    document.body.classList.add(currentTheme);

    tabs.forEach(tab => {
        tab.addEventListener("click", () => {
            const target = tab.dataset.tab;

            tabs.forEach(t => t.classList.remove("active"));
            tab.classList.add("active");

            sections.forEach(section => {
                if (section.id === target) {
                    section.classList.add("active");
                } else {
                    section.classList.remove("active");
                }
            });
        });
    });

    navigateButtons.forEach(button => {
        button.addEventListener("click", () => {
            const target = button.dataset.tab;
            tabs.forEach(tab => {
                if (tab.dataset.tab === target) {
                    tab.click();
                }
            });
        });
    });

    predictCropButton.addEventListener("click", () => {
        const selectedDistrict = agriculturalAreas.value;
        fetch('/predict_crop_recommendations', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ district: selectedDistrict })
        })
        .then(response => response.json())
        .then(data => {
            cropRecommendations.innerHTML = `<h2>Recommended Crops for ${selectedDistrict}:</h2>
                                             <ul>${data.recommendations.map(crop => `<li>${crop}</li>`).join('')}</ul>`;
        })
        .catch(error => {
            console.error('Error:', error);
        });
    });

    predictPriceButton.addEventListener("click", () => {
        const year = yearInput.value;
        const month = monthInput.value;
        const week = weekInput.value;
        const crop = cropInput.value; // Get the crop value

        fetch('/predict_price/', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ crop: crop, year: year, month: month, week: week }) // Include crop in the request body
        })
        .then(response => response.json())
        .then(data => {
            pricePrediction.innerHTML = `<h2>Predicted Price:</h2><p>${data.predicted_price}</p>`;
        })
        .catch(error => {
            console.error('Error:', error);
        });
    });

    sendBtn.addEventListener('click', sendMessage);
    userInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter') sendMessage();
    });

    themeToggleBtn.addEventListener('click', toggleTheme);

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
        })
        .catch(error => {
            console.error('Error:', error);
        });
    });

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
        })
        .catch(error => {
            console.error('Error:', error);
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

    function generateBotResponse(userMessage) {
        // Simple mock response for demonstration
        return `You said: "${userMessage}"`;
    }

    window.sendPrompt = function (prompt) {
        userInput.value = prompt;
        sendMessage();
    };

    function toggleTheme() {
        document.body.classList.toggle('light');
        document.body.classList.toggle('dark');
        currentTheme = document.body.classList.contains('light') ? 'light' : 'dark';
        localStorage.setItem('theme', currentTheme);
    }

    loadThreads();
    loadChatHistory(currentThreadId);
});

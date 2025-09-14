/**
 * Chat Integration for A/B Test Designer
 * Add this script to your main index.html to integrate the chat assistant
 */

class ChatIntegration {
    constructor() {
        this.sessionId = null;
        this.isInitialized = false;
        this.chatWidget = null;
        
        this.initializeChat();
    }
    
    async initializeChat() {
        try {
            // Start new chat session
            const response = await fetch('/api/chat/start', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                }
            });
            
            const data = await response.json();
            
            if (data.success) {
                this.sessionId = data.session_id;
                this.isInitialized = true;
                
                // Initialize chat widget if not already present
                if (!document.getElementById('chatWidget')) {
                    this.injectChatWidget();
                }
                
                console.log('Chat assistant initialized successfully');
            } else {
                console.error('Failed to initialize chat:', data.error);
            }
            
        } catch (error) {
            console.error('Error initializing chat:', error);
        }
    }
    
    injectChatWidget() {
        // Create chat widget HTML
        const chatHTML = `
            <!-- Chat Toggle Button -->
            <button class="chat-toggle-btn" id="chatToggle" onclick="window.chatIntegration.toggleChat()">💬</button>

            <!-- Chat Widget -->
            <div class="chat-widget minimized" id="chatWidget">
                <div class="chat-header" onclick="window.chatIntegration.toggleChat()">
                    <h3>
                        <span class="chat-status"></span>
                        AI Assistant
                    </h3>
                    <button class="minimize-btn" onclick="window.chatIntegration.toggleChat()" id="minimizeBtn">+</button>
                </div>

                <div class="chat-messages" id="chatMessages">
                    <!-- Messages will be added here dynamically -->
                </div>

                <div class="suggestions-container" id="suggestionsContainer">
                    <div class="suggestions" id="suggestions">
                        <!-- Suggestion chips will be added here -->
                    </div>
                </div>

                <div class="chat-input-container">
                    <div class="chat-input-wrapper">
                        <textarea 
                            class="chat-input" 
                            id="chatInput" 
                            placeholder="Ask me about A/B testing..."
                            rows="1"
                            onkeydown="window.chatIntegration.handleKeyPress(event)"
                        ></textarea>
                        <button class="send-btn" id="sendBtn" onclick="window.chatIntegration.sendMessage()">
                            ➤
                        </button>
                    </div>
                </div>
            </div>
        `;
        
        // Inject CSS styles
        const chatCSS = `
            <style>
                /* Chat Widget Styles - Same as in chat_interface.html */
                .chat-widget {
                    position: fixed;
                    bottom: 20px;
                    right: 20px;
                    width: 380px;
                    height: 600px;
                    background: white;
                    border-radius: 20px;
                    box-shadow: 0 20px 40px rgba(0, 0, 0, 0.15);
                    display: flex;
                    flex-direction: column;
                    overflow: hidden;
                    z-index: 1000;
                    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                    transition: all 0.3s ease;
                }

                .chat-widget.minimized {
                    height: 70px;
                    width: 320px;
                }

                .chat-header {
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    color: white;
                    padding: 20px;
                    display: flex;
                    align-items: center;
                    justify-content: space-between;
                    cursor: pointer;
                }

                .chat-header h3 {
                    margin: 0;
                    font-size: 1.2em;
                    display: flex;
                    align-items: center;
                    gap: 10px;
                }

                .chat-status {
                    width: 8px;
                    height: 8px;
                    background: #4ade80;
                    border-radius: 50%;
                    animation: pulse 2s infinite;
                }

                @keyframes pulse {
                    0%, 100% { opacity: 1; }
                    50% { opacity: 0.5; }
                }

                .minimize-btn {
                    background: none;
                    border: none;
                    color: white;
                    font-size: 1.5em;
                    cursor: pointer;
                    padding: 5px;
                    border-radius: 5px;
                    transition: background 0.2s;
                }

                .minimize-btn:hover {
                    background: rgba(255, 255, 255, 0.2);
                }

                .chat-messages {
                    flex: 1;
                    padding: 20px;
                    overflow-y: auto;
                    background: #f8f9fa;
                    display: flex;
                    flex-direction: column;
                    gap: 15px;
                }

                .chat-widget.minimized .chat-messages,
                .chat-widget.minimized .chat-input-container,
                .chat-widget.minimized .suggestions-container {
                    display: none;
                }

                .message {
                    max-width: 85%;
                    padding: 12px 16px;
                    border-radius: 18px;
                    line-height: 1.4;
                    animation: messageIn 0.3s ease;
                }

                @keyframes messageIn {
                    from { opacity: 0; transform: translateY(10px); }
                    to { opacity: 1; transform: translateY(0); }
                }

                .message.user {
                    background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
                    color: white;
                    align-self: flex-end;
                    margin-left: auto;
                }

                .message.assistant {
                    background: white;
                    color: #333;
                    border: 1px solid #e1e5e9;
                    align-self: flex-start;
                    position: relative;
                }

                .message.assistant::before {
                    content: '🤖';
                    position: absolute;
                    left: -30px;
                    top: 50%;
                    transform: translateY(-50%);
                    font-size: 1.2em;
                }

                .suggestions-container {
                    padding: 0 20px 15px;
                    background: #f8f9fa;
                }

                .suggestions {
                    display: flex;
                    flex-wrap: wrap;
                    gap: 8px;
                }

                .suggestion-chip {
                    background: white;
                    border: 1px solid #667eea;
                    color: #667eea;
                    padding: 8px 12px;
                    border-radius: 20px;
                    font-size: 0.9em;
                    cursor: pointer;
                    transition: all 0.2s ease;
                    white-space: nowrap;
                }

                .suggestion-chip:hover {
                    background: #667eea;
                    color: white;
                    transform: translateY(-1px);
                }

                .chat-input-container {
                    padding: 20px;
                    background: white;
                    border-top: 1px solid #e1e5e9;
                }

                .chat-input-wrapper {
                    display: flex;
                    gap: 10px;
                    align-items: flex-end;
                }

                .chat-input {
                    flex: 1;
                    border: 2px solid #e1e5e9;
                    border-radius: 25px;
                    padding: 12px 20px;
                    font-size: 14px;
                    resize: none;
                    min-height: 20px;
                    max-height: 100px;
                    font-family: inherit;
                    transition: border-color 0.2s;
                }

                .chat-input:focus {
                    outline: none;
                    border-color: #667eea;
                }

                .send-btn {
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    color: white;
                    border: none;
                    width: 45px;
                    height: 45px;
                    border-radius: 50%;
                    cursor: pointer;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    transition: all 0.2s ease;
                    font-size: 1.2em;
                }

                .send-btn:hover {
                    transform: scale(1.05);
                }

                .send-btn:disabled {
                    opacity: 0.5;
                    cursor: not-allowed;
                    transform: none;
                }

                .chat-toggle-btn {
                    position: fixed;
                    bottom: 20px;
                    right: 20px;
                    width: 60px;
                    height: 60px;
                    border-radius: 50%;
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    color: white;
                    border: none;
                    font-size: 1.5em;
                    cursor: pointer;
                    box-shadow: 0 8px 25px rgba(102, 126, 234, 0.3);
                    transition: all 0.3s ease;
                    z-index: 999;
                }

                .chat-toggle-btn:hover {
                    transform: scale(1.1);
                    box-shadow: 0 12px 35px rgba(102, 126, 234, 0.4);
                }

                .chat-toggle-btn.hidden {
                    display: none;
                }

                .analysis-ready {
                    background: linear-gradient(135deg, #4ade80 0%, #22c55e 100%);
                    color: white;
                    padding: 15px;
                    border-radius: 12px;
                    margin: 10px 0;
                    text-align: center;
                    font-weight: 600;
                }

                .launch-test-btn {
                    background: linear-gradient(135deg, #f59e0b 0%, #d97706 100%);
                    color: white;
                    border: none;
                    padding: 12px 20px;
                    border-radius: 8px;
                    cursor: pointer;
                    font-weight: 600;
                    width: 100%;
                    margin-top: 10px;
                    transition: transform 0.2s;
                }

                .launch-test-btn:hover {
                    transform: translateY(-2px);
                }

                /* Mobile responsive */
                @media (max-width: 480px) {
                    .chat-widget {
                        width: calc(100vw - 20px);
                        height: calc(100vh - 40px);
                        bottom: 10px;
                        right: 10px;
                        left: 10px;
                        border-radius: 15px;
                    }

                    .chat-widget.minimized {
                        height: 60px;
                        width: calc(100vw - 40px);
                    }
                }
            </style>
        `;
        
        // Add CSS to head
        document.head.insertAdjacentHTML('beforeend', chatCSS);
        
        // Add HTML to body
        document.body.insertAdjacentHTML('beforeend', chatHTML);
        
        // Initialize chat state
        this.isMinimized = true;
        this.showInitialMessage();
    }
    
    showInitialMessage() {
        this.addMessage('assistant', 
            "👋 Hi! I'm your A/B Test Designer assistant. I'll help you set up and analyze your experiments!\n\nWhat would you like to do today?"
        );
        
        this.showSuggestions([
            'Design new A/B test',
            'Analyze test results',
            'Get sample size help',
            'Test ideas'
        ]);
    }
    
    toggleChat() {
        const widget = document.getElementById('chatWidget');
        const toggleBtn = document.getElementById('chatToggle');
        const minimizeBtn = document.getElementById('minimizeBtn');

        this.isMinimized = !this.isMinimized;
        
        if (this.isMinimized) {
            widget.classList.add('minimized');
            toggleBtn.classList.remove('hidden');
            minimizeBtn.textContent = '+';
        } else {
            widget.classList.remove('minimized');
            toggleBtn.classList.add('hidden');
            minimizeBtn.textContent = '−';
            this.scrollToBottom();
            document.getElementById('chatInput').focus();
        }
    }
    
    handleKeyPress(event) {
        if (event.key === 'Enter' && !event.shiftKey) {
            event.preventDefault();
            this.sendMessage();
        }
    }
    
    async sendMessage() {
        const input = document.getElementById('chatInput');
        const message = input.value.trim();
        
        if (!message || !this.sessionId) return;

        // Add user message to UI
        this.addMessage('user', message);
        input.value = '';
        input.style.height = 'auto';

        // Show typing indicator
        this.showTyping();

        try {
            // Send message to backend
            const response = await fetch(`/api/chat/${this.sessionId}/message`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ message })
            });

            const data = await response.json();
            
            // Hide typing indicator
            this.hideTyping();
            
            if (data.success) {
                // Add assistant response
                this.addMessage('assistant', data.message);
                
                // Show suggestions if any
                if (data.suggestions && data.suggestions.length > 0) {
                    this.showSuggestions(data.suggestions);
                } else {
                    this.showSuggestions([]);
                }
                
                // Handle analysis ready state
                if (data.analysis_ready && data.parameters) {
                    this.showAnalysisReady(data.parameters);
                }
                
            } else {
                this.addMessage('assistant', `Sorry, I encountered an error: ${data.error}`);
            }
            
        } catch (error) {
            this.hideTyping();
            this.addMessage('assistant', 'Sorry, I\'m having trouble connecting. Please try again.');
            console.error('Chat error:', error);
        }
    }
    
    addMessage(sender, text) {
        const messagesContainer = document.getElementById('chatMessages');
        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${sender}`;
        
        // Format text with basic markdown support
        const formattedText = this.formatMessage(text);
        messageDiv.innerHTML = formattedText;

        messagesContainer.appendChild(messageDiv);
        this.scrollToBottom();
    }
    
    formatMessage(text) {
        return text
            .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
            .replace(/\*(.*?)\*/g, '<em>$1</em>')
            .replace(/\n/g, '<br>')
            .replace(/• /g, '• ');
    }
    
    showTyping() {
        const messagesContainer = document.getElementById('chatMessages');
        const typingDiv = document.createElement('div');
        typingDiv.className = 'typing-indicator';
        typingDiv.id = 'typingIndicator';
        typingDiv.innerHTML = `
            <span>AI is thinking</span>
            <div class="typing-dots">
                <div class="typing-dot"></div>
                <div class="typing-dot"></div>
                <div class="typing-dot"></div>
            </div>
        `;
        
        messagesContainer.appendChild(typingDiv);
        this.scrollToBottom();
    }
    
    hideTyping() {
        const typingIndicator = document.getElementById('typingIndicator');
        if (typingIndicator) {
            typingIndicator.remove();
        }
    }
    
    showSuggestions(suggestions) {
        const suggestionsContainer = document.getElementById('suggestions');
        suggestionsContainer.innerHTML = '';

        suggestions.forEach(suggestion => {
            const chip = document.createElement('div');
            chip.className = 'suggestion-chip';
            chip.textContent = suggestion;
            chip.onclick = () => {
                document.getElementById('chatInput').value = suggestion;
                this.sendMessage();
            };
            suggestionsContainer.appendChild(chip);
        });

        document.getElementById('suggestionsContainer').style.display = 
            suggestions.length > 0 ? 'block' : 'none';
    }
    
    showAnalysisReady(parameters) {
        const messagesContainer = document.getElementById('chatMessages');
        const readyDiv = document.createElement('div');
        readyDiv.className = 'analysis-ready';
        readyDiv.innerHTML = `
            🎯 Your test parameters are ready!
            <button class="launch-test-btn" onclick="window.chatIntegration.launchMainInterface()">
                🚀 Launch A/B Test Designer
            </button>
        `;
        messagesContainer.appendChild(readyDiv);
        this.scrollToBottom();
        
        // Store parameters for main form
        this.lastParameters = parameters;
    }
    
    launchMainInterface() {
        if (this.lastParameters) {
            // Fill the main form with chat data
            this.fillFormWithChatData(this.lastParameters);
            
            // Minimize chat
            if (!this.isMinimized) {
                this.toggleChat();
            }
            
            // Show success message
            this.showSuccessNotification();
        }
    }
    
    fillFormWithChatData(params) {
        // Fill the main A/B Test Designer form with parameters from chat
        const questionInput = document.getElementById('research-question');
        const hypothesisInput = document.getElementById('hypothesis');
        const metricSelect = document.getElementById('metric');
        const effectInput = document.getElementById('expected-effect');
        const baselineInput = document.getElementById('baseline-rate');
        
        if (questionInput) questionInput.value = params.question || '';
        if (hypothesisInput) hypothesisInput.value = params.hypothesis || '';
        if (metricSelect) metricSelect.value = params.metric || '';
        if (effectInput) effectInput.value = params.effect_size || '';
        if (baselineInput) baselineInput.value = params.baseline_rate || '';
        
        // Trigger the main form's design experiment function
        const designBtn = document.getElementById('designBtn');
        if (designBtn) {
            designBtn.click();
        }
        
        console.log('Form filled with chat parameters:', params);
    }
    
    showSuccessNotification() {
        // Create success notification
        const notification = document.createElement('div');
        notification.style.cssText = `
            position: fixed;
            top: 20px;
            right: 20px;
            background: linear-gradient(135deg, #4ade80 0%, #22c55e 100%);
            color: white;
            padding: 15px 20px;
            border-radius: 10px;
            box-shadow: 0 10px 25px rgba(74, 222, 128, 0.3);
            z-index: 10000;
            animation: slideInRight 0.3s ease;
        `;
        notification.innerHTML = '✅ Chat parameters applied to main form!';
        
        document.body.appendChild(notification);
        
        // Remove after 3 seconds
        setTimeout(() => {
            notification.remove();
        }, 3000);
    }
    
    scrollToBottom() {
        const messagesContainer = document.getElementById('chatMessages');
        messagesContainer.scrollTop = messagesContainer.scrollHeight;
    }
    
    async resetChat() {
        if (!this.sessionId) return;
        
        try {
            const response = await fetch(`/api/chat/${this.sessionId}/reset`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                }
            });
            
            const data = await response.json();
            
            if (data.success) {
                // Clear messages
                document.getElementById('chatMessages').innerHTML = '';
                
                // Show fresh greeting
                this.addMessage('assistant', data.message);
                this.showSuggestions(data.suggestions || []);
            }
            
        } catch (error) {
            console.error('Error resetting chat:', error);
        }
    }
}

// Initialize chat integration when DOM is loaded
document.addEventListener('DOMContentLoaded', function() {
    // Only initialize if not already present
    if (!window.chatIntegration) {
        window.chatIntegration = new ChatIntegration();
    }
});

// Export for global access
window.ChatIntegration = ChatIntegration;
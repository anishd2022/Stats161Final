// Chat interface JavaScript

const chatContainer = document.getElementById('chatContainer');
const queryInput = document.getElementById('queryInput');
const sendButton = document.getElementById('sendButton');

// Auto-resize textarea
queryInput.addEventListener('input', function() {
    this.style.height = 'auto';
    this.style.height = this.scrollHeight + 'px';
    
    // Enable/disable send button
    sendButton.disabled = !this.value.trim();
});

// Handle Enter key (send) vs Shift+Enter (new line)
queryInput.addEventListener('keydown', function(e) {
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        if (!sendButton.disabled) {
            sendMessage();
        }
    }
});

// Send button click
sendButton.addEventListener('click', sendMessage);

// Example query click handlers
document.addEventListener('DOMContentLoaded', function() {
    const exampleQueries = document.querySelectorAll('.example-queries li');
    exampleQueries.forEach(li => {
        li.addEventListener('click', function() {
            queryInput.value = this.textContent.trim();
            queryInput.style.height = 'auto';
            queryInput.style.height = queryInput.scrollHeight + 'px';
            sendButton.disabled = false;
            queryInput.focus();
        });
    });
});

function sendMessage() {
    const query = queryInput.value.trim();
    if (!query) return;
    
    // Clear input
    queryInput.value = '';
    queryInput.style.height = 'auto';
    sendButton.disabled = true;
    
    // Remove welcome message if present
    const welcomeMessage = document.querySelector('.welcome-message');
    if (welcomeMessage) {
        welcomeMessage.remove();
    }
    
    // Add user message
    addMessage('user', query);
    
    // Show loading indicator
    const loadingId = addMessage('bot', '', true);
    
    // Send query to backend
    fetch('/api/query', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({ query: query })
    })
    .then(response => response.json())
    .then(data => {
        // Remove loading indicator
        const loadingElement = document.getElementById(loadingId);
        if (loadingElement) {
            loadingElement.remove();
        }
        
        if (data.success) {
            // Add bot response
            let content = data.answer;
            
            // Add SQL queries if they were executed
            if (data.needs_sql && data.sql_queries && data.sql_queries.length > 0) {
                content += '\n\n<div class="sql-queries">';
                content += '<div class="sql-queries-title">SQL Queries Executed:</div>';
                data.sql_queries.forEach((q, idx) => {
                    content += `<div class="sql-query">${escapeHtml(q.query)}</div>`;
                    if (q.row_count !== undefined) {
                        content += `<div style="font-size: 0.8rem; color: var(--text-secondary); margin-top: 0.25rem; margin-bottom: 0.5rem;">${q.row_count} row(s) returned</div>`;
                    }
                });
                content += '</div>';
            }
            
            addMessage('bot', content);
        } else {
            // Show error
            addMessage('error', `Error: ${data.error || 'Something went wrong'}`);
        }
    })
    .catch(error => {
        // Remove loading indicator
        const loadingElement = document.getElementById(loadingId);
        if (loadingElement) {
            loadingElement.remove();
        }
        
        // Show error
        addMessage('error', `Error: ${error.message || 'Failed to connect to server'}`);
    });
}

function addMessage(type, content, isLoading = false) {
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${type}`;
    
    if (isLoading) {
        const uniqueId = 'loading-' + Date.now();
        messageDiv.id = uniqueId;
        messageDiv.innerHTML = `
            <div class="message-header">
                <span class="icon">🤖</span>
                <span>Assistant</span>
            </div>
            <div class="message-content">
                <div class="loading"></div>
                <span style="margin-left: 0.5rem;">Thinking...</span>
            </div>
        `;
        chatContainer.appendChild(messageDiv);
        scrollToBottom();
        return uniqueId;
    }
    
    const icon = type === 'user' ? '👤' : type === 'error' ? '⚠️' : '🤖';
    const header = type === 'user' ? 'You' : type === 'error' ? 'Error' : 'Assistant';
    
    messageDiv.innerHTML = `
        <div class="message-header">
            <span class="icon">${icon}</span>
            <span>${header}</span>
        </div>
        <div class="message-content">${formatContent(content)}</div>
    `;
    
    chatContainer.appendChild(messageDiv);
    scrollToBottom();
    
    return messageDiv.id;
}

function formatContent(content) {
    // Convert markdown-style code blocks to HTML
    content = escapeHtml(content);
    
    // Handle code blocks
    content = content.replace(/```(\w+)?\n([\s\S]*?)```/g, function(match, lang, code) {
        return `<pre><code>${code.trim()}</code></pre>`;
    });
    
    // Handle inline code
    content = content.replace(/`([^`]+)`/g, '<code>$1</code>');
    
    // Handle line breaks
    content = content.replace(/\n/g, '<br>');
    
    // Handle SQL queries div (already HTML)
    if (content.includes('<div class="sql-queries">')) {
        // Don't escape HTML inside SQL queries div
        const parts = content.split(/(<div class="sql-queries">[\s\S]*?<\/div>)/);
        let result = '';
        for (let i = 0; i < parts.length; i++) {
            if (parts[i].startsWith('<div class="sql-queries">')) {
                result += parts[i];
            } else {
                result += parts[i];
            }
        }
        return result;
    }
    
    return content;
}

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

function scrollToBottom() {
    chatContainer.scrollTop = chatContainer.scrollHeight;
}

// Focus input on load
window.addEventListener('load', () => {
    queryInput.focus();
});


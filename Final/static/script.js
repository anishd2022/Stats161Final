// Chat interface JavaScript

const chatContainer = document.getElementById('chatContainer');
const queryInput = document.getElementById('queryInput');
const sendButton = document.getElementById('sendButton');
const newChatButton = document.getElementById('newChatButton');
const chatList = document.getElementById('chatList');

// Current session state
let currentSessionId = null;
let chatMessages = {}; // session_id -> array of messages

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

// Initialize on page load
document.addEventListener('DOMContentLoaded', function() {
    loadChatList();
    
    // Example query click handlers
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
    
    // New chat button handler
    newChatButton.addEventListener('click', createNewChat);
});

function loadChatList() {
    fetch('/api/chats')
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                renderChatList(data.chats);
                // If no current session and chats exist, select the first one
                if (!currentSessionId && data.chats.length > 0) {
                    switchToChat(data.chats[0].session_id);
                } else if (currentSessionId) {
                    // Re-render to update active state without switching
                    renderChatList(data.chats);
                }
            }
        })
        .catch(error => {
            console.error('Error loading chat list:', error);
        });
}

function renderChatList(chats) {
    chatList.innerHTML = '';
    chats.forEach(chat => {
        const chatItem = document.createElement('div');
        chatItem.className = 'chat-item';
        if (chat.session_id === currentSessionId) {
            chatItem.classList.add('active');
        }
        chatItem.textContent = chat.title;
        chatItem.addEventListener('click', () => switchToChat(chat.session_id));
        chatList.appendChild(chatItem);
    });
}

function createNewChat() {
    return fetch('/api/chats/new', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        }
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            currentSessionId = data.session_id;
            chatMessages[currentSessionId] = [];
            clearChatContainer();
            loadChatList();
            queryInput.focus();
            return data;
        } else {
            throw new Error(data.error || 'Failed to create chat');
        }
    })
    .catch(error => {
        console.error('Error creating new chat:', error);
        throw error;
    });
}

function switchToChat(sessionId) {
    currentSessionId = sessionId;
    
    // Clear current chat display
    clearChatContainer();
    
    // Load messages for this session if they exist
    if (chatMessages[sessionId] && chatMessages[sessionId].length > 0) {
        chatMessages[sessionId].forEach(msg => {
            addMessageToContainer(msg.type, msg.content, false, msg.sqlQueriesHtml, msg.tableData);
        });
    } else {
        // Show welcome message if no messages
        showWelcomeMessage();
    }
    
    // Re-render chat list to update active state
    fetch('/api/chats')
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                renderChatList(data.chats);
            }
        })
        .catch(error => {
            console.error('Error loading chat list:', error);
        });
    
    queryInput.focus();
}

function clearChatContainer() {
    chatContainer.innerHTML = '';
}

function showWelcomeMessage() {
    chatContainer.innerHTML = `
        <div class="welcome-message">
            <div class="welcome-icon">💬</div>
            <h2>Welcome!</h2>
            <p>I'm your database assistant. Ask me anything about your data, and I'll help you find the answers.</p>
            <p class="examples">Try asking things like:</p>
            <ul class="example-queries">
                <li>"How many users are in the database?"</li>
                <li>"What is the average age of users?"</li>
                <li>"Show me the top 10 most active users"</li>
            </ul>
        </div>
    `;
    
    // Re-attach example query handlers
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
}

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
    
    // Initialize session if needed
    if (!currentSessionId) {
        createNewChat().then(() => {
            // After creating chat, send the message
            sendMessageWithSession(query);
        });
        return;
    }
    
    sendMessageWithSession(query);
}

function sendMessageWithSession(query) {
    // Initialize messages array for this session if needed
    if (!chatMessages[currentSessionId]) {
        chatMessages[currentSessionId] = [];
    }
    
    // Add user message to display and storage
    addMessage('user', query);
    chatMessages[currentSessionId].push({
        type: 'user',
        content: query,
        sqlQueriesHtml: null
    });
    
    // Show loading indicator
    const loadingId = addMessage('bot', '', true);
    
    // Send query to backend with session_id
    fetch('/api/query', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({ 
            query: query,
            session_id: currentSessionId
        })
    })
    .then(response => response.json())
    .then(data => {
        // Remove loading indicator
        const loadingElement = document.getElementById(loadingId);
        if (loadingElement) {
            loadingElement.remove();
        }
        
        // Update session_id if returned (for new chats)
        if (data.session_id) {
            currentSessionId = data.session_id;
        }
        
        if (data.success) {
            // Strip any HTML from the LLM response (keep only plain text)
            let content = stripHtml(data.answer);
            
            // Add SQL queries if they were executed (as separate HTML element)
            let sqlQueriesHtml = null;
            if (data.needs_sql && data.sql_queries && data.sql_queries.length > 0) {
                sqlQueriesHtml = formatSqlQueries(data.sql_queries);
            }
            
            // Get table data if available
            let tableData = null;
            if (data.table_data && data.table_data.length > 0) {
                tableData = data.table_data;
            }
            
            // Add message with table data
            addMessage('bot', content, false, sqlQueriesHtml, tableData);
            
            // Store bot message
            chatMessages[currentSessionId].push({
                type: 'bot',
                content: content,
                sqlQueriesHtml: sqlQueriesHtml,
                tableData: tableData
            });
            
            // Reload chat list to update titles
            loadChatList();
        } else {
            // Show error
            addMessage('error', `Error: ${data.error || 'Something went wrong'}`);
            chatMessages[currentSessionId].push({
                type: 'error',
                content: `Error: ${data.error || 'Something went wrong'}`,
                sqlQueriesHtml: null
            });
        }
    })
    .catch(error => {
        // Remove loading indicator
        const loadingElement = document.getElementById(loadingId);
        if (loadingElement) {
            loadingElement.remove();
        }
        
        // Show error
        const errorMsg = `Error: ${error.message || 'Failed to connect to server'}`;
        addMessage('error', errorMsg);
        if (currentSessionId && chatMessages[currentSessionId]) {
            chatMessages[currentSessionId].push({
                type: 'error',
                content: errorMsg,
                sqlQueriesHtml: null
            });
        }
    });
}

// Threshold for displaying tables (if more rows, only show download button)
const TABLE_DISPLAY_THRESHOLD = 100;

function renderTableData(tableDataArray) {
    let html = '';
    const timestamp = Date.now();
    
    // Store table data globally for download access
    if (!window.tableDataCache) {
        window.tableDataCache = {};
    }
    
    tableDataArray.forEach((tableData, tableIndex) => {
        const { columns, data, row_count } = tableData;
        
        if (!columns || columns.length === 0 || !data || data.length === 0) {
            return; // Skip empty tables
        }
        
        const shouldDisplayTable = row_count <= TABLE_DISPLAY_THRESHOLD;
        const tableId = `table-${timestamp}-${tableIndex}`;
        
        // Store table data for download access
        window.tableDataCache[tableId] = { columns, data, row_count };
        
        html += '<div class="data-table-container">';
        
        // Display table if small enough
        if (shouldDisplayTable) {
            html += '<div class="data-table-wrapper">';
            html += '<table class="data-table">';
            
            // Header
            html += '<thead><tr>';
            columns.forEach(col => {
                html += `<th>${escapeHtml(String(col))}</th>`;
            });
            html += '</tr></thead>';
            
            // Body
            html += '<tbody>';
            data.forEach(row => {
                html += '<tr>';
                columns.forEach(col => {
                    const value = row[col] !== undefined && row[col] !== null ? String(row[col]) : '';
                    html += `<td>${escapeHtml(value)}</td>`;
                });
                html += '</tr>';
            });
            html += '</tbody>';
            
            html += '</table>';
            html += '</div>';
        } else {
            // For large tables, show a message instead of the table
            html += `<div class="table-info-message">Table contains ${row_count} rows (too large to display)</div>`;
        }
        
        // Download CSV button (always shown)
        html += `<div class="table-actions">`;
        html += `<button class="download-csv-btn" onclick="downloadTableAsCSV('${tableId}')">`;
        html += `<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">`;
        html += `<path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"></path>`;
        html += `<polyline points="7 10 12 15 17 10"></polyline>`;
        html += `<line x1="12" y1="15" x2="12" y2="3"></line>`;
        html += `</svg>`;
        html += `Download CSV (${row_count} rows)`;
        html += `</button>`;
        html += `</div>`;
        
        html += '</div>'; // Close data-table-container
    });
    
    return html;
}

// Make downloadTableAsCSV globally accessible
window.downloadTableAsCSV = function(tableId) {
    const tableData = window.tableDataCache && window.tableDataCache[tableId];
    if (!tableData) {
        console.error('Table data not found for download');
        return;
    }
    
    const { columns, data } = tableData;
    
    // Create CSV content
    let csvContent = '';
    
    // Header row
    csvContent += columns.map(col => escapeCSVValue(String(col))).join(',') + '\n';
    
    // Data rows
    data.forEach(row => {
        const rowValues = columns.map(col => {
            const value = row[col] !== undefined && row[col] !== null ? String(row[col]) : '';
            return escapeCSVValue(value);
        });
        csvContent += rowValues.join(',') + '\n';
    });
    
    // Create blob and download
    const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
    const link = document.createElement('a');
    const url = URL.createObjectURL(blob);
    link.setAttribute('href', url);
    link.setAttribute('download', `table_data_${Date.now()}.csv`);
    link.style.visibility = 'hidden';
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);
};

function escapeCSVValue(value) {
    // If value contains comma, quote, or newline, wrap in quotes and escape quotes
    if (value.includes(',') || value.includes('"') || value.includes('\n') || value.includes('\r')) {
        return '"' + value.replace(/"/g, '""') + '"';
    }
    return value;
}

function addMessage(type, content, isLoading = false, sqlQueriesHtml = null, tableData = null) {
    return addMessageToContainer(type, content, isLoading, sqlQueriesHtml, tableData);
}

function addMessageToContainer(type, content, isLoading = false, sqlQueriesHtml = null, tableData = null) {
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
    
    // If we have table_data, skip markdown table rendering in text to avoid duplicates
    const skipMarkdownTables = tableData && tableData.length > 0;
    let contentHtml = formatContent(content, skipMarkdownTables);
    if (sqlQueriesHtml) {
        contentHtml += sqlQueriesHtml;
    }
    
    // Add table data rendering if available
    let tableHtml = '';
    if (tableData && tableData.length > 0) {
        tableHtml = renderTableData(tableData);
    }
    
    messageDiv.innerHTML = `
        <div class="message-header">
            <span class="icon">${icon}</span>
            <span>${header}</span>
        </div>
        <div class="message-content">${contentHtml}${tableHtml}</div>
    `;
    
    chatContainer.appendChild(messageDiv);
    scrollToBottom();
    
    return messageDiv.id;
}

function formatContent(content, skipMarkdownTables = false) {
    // Use placeholders to preserve tables during HTML escaping
    const tablePlaceholders = [];
    let placeholderIndex = 0;
    
    // Convert markdown tables to HTML and replace with placeholders
    // If skipMarkdownTables is true, remove markdown tables instead of converting them
    content = content.replace(/(\|.+\|\r?\n\|[-\s|:]+\|\r?\n(?:\|.+\|\r?\n?)+)/g, function(match) {
        if (skipMarkdownTables) {
            // Remove markdown tables when we have table_data to avoid duplicates
            return '';
        } else {
            const tableHtml = convertMarkdownTable(match);
            const placeholder = `__TABLE_PLACEHOLDER_${placeholderIndex}__`;
            tablePlaceholders[placeholderIndex] = tableHtml;
            placeholderIndex++;
            return placeholder;
        }
    });
    
    // Now escape HTML (tables are safe in placeholders)
    content = escapeHtml(content);
    
    // Restore tables from placeholders (only if we didn't skip them)
    if (!skipMarkdownTables) {
        tablePlaceholders.forEach((tableHtml, index) => {
            content = content.replace(`__TABLE_PLACEHOLDER_${index}__`, tableHtml);
        });
    }
    
    // Handle code blocks
    content = content.replace(/```(\w+)?\n([\s\S]*?)```/g, function(match, lang, code) {
        return `<pre><code>${escapeHtml(code.trim())}</code></pre>`;
    });
    
    // Handle inline code
    content = content.replace(/`([^`]+)`/g, '<code>$1</code>');
    
    // Handle line breaks
    content = content.replace(/\n/g, '<br>');
    
    return content;
}

function convertMarkdownTable(match) {
    const lines = match.trim().split(/\r?\n/);
    if (lines.length < 2) return match; // Need at least header and separator
    
    // Parse header row
    const headerRow = lines[0];
    const headers = headerRow.split('|').map(h => h.trim()).filter(h => h);
    
    if (headers.length === 0) return match; // Invalid table
    
    // Skip separator row (lines[1])
    // Parse data rows
    const dataRows = lines.slice(2).map(row => {
        const cells = row.split('|').map(cell => cell.trim());
        // Filter out empty cells at start/end (from leading/trailing |)
        return cells.filter((cell, idx) => idx > 0 && idx <= headers.length);
    }).filter(row => row.length > 0);
    
    // Build HTML table
    let html = '<table class="markdown-table">';
    
    // Header
    html += '<thead><tr>';
    headers.forEach(header => {
        html += `<th>${escapeHtml(header)}</th>`;
    });
    html += '</tr></thead>';
    
    // Body
    if (dataRows.length > 0) {
        html += '<tbody>';
        dataRows.forEach(row => {
            html += '<tr>';
            headers.forEach((_, idx) => {
                const cell = escapeHtml(row[idx] || '');
                html += `<td>${cell}</td>`;
            });
            html += '</tr>';
        });
        html += '</tbody>';
    }
    
    html += '</table>';
    
    return html;
}

function stripHtml(html) {
    // Remove all HTML tags from the response, keeping only text content
    const tmp = document.createElement('div');
    tmp.innerHTML = html;
    return tmp.textContent || tmp.innerText || '';
}

function formatSqlQueries(sqlQueries) {
    let html = '<div class="sql-queries">';
    html += '<div class="sql-queries-title">SQL Queries Executed:</div>';
    sqlQueries.forEach((q, idx) => {
        html += `<div class="sql-query">${escapeHtml(q.query)}</div>`;
        if (q.row_count !== undefined) {
            const countText = typeof q.row_count === 'number' ? `${q.row_count} row(s)` : 'rows';
            html += `<div style="font-size: 0.8rem; color: var(--text-secondary); margin-top: 0.25rem; margin-bottom: 0.5rem;">${countText} returned</div>`;
        }
    });
    html += '</div>';
    return html;
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


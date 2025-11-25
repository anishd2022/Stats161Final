// Chat interface JavaScript

const chatContainer = document.getElementById('chatContainer');
const queryInput = document.getElementById('queryInput');
const sendButton = document.getElementById('sendButton');
const newChatButton = document.getElementById('newChatButton');
const chatList = document.getElementById('chatList');
const downloadPdfButton = document.getElementById('downloadPdfButton');

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
    updateDownloadButtonState();
    
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

    if (downloadPdfButton) {
        downloadPdfButton.addEventListener('click', downloadCurrentChatAsPDF);
    }
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
            updateDownloadButtonState();
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
    updateDownloadButtonState();
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
    updateDownloadButtonState();
    
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
            
            // Get chart config if available
            let chartConfig = null;
            if (data.chart_config) {
                chartConfig = data.chart_config;
            }
            
            // Add message with table data and chart
            addMessage('bot', content, false, sqlQueriesHtml, tableData, chartConfig);
            
            // Store bot message
            chatMessages[currentSessionId].push({
                type: 'bot',
                content: content,
                sqlQueriesHtml: sqlQueriesHtml,
                tableData: tableData,
                chartConfig: chartConfig
            });
            updateDownloadButtonState();
            
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
            updateDownloadButtonState();
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
            updateDownloadButtonState();
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

function addMessage(type, content, isLoading = false, sqlQueriesHtml = null, tableData = null, chartConfig = null) {
    return addMessageToContainer(type, content, isLoading, sqlQueriesHtml, tableData, chartConfig);
}

function addMessageToContainer(type, content, isLoading = false, sqlQueriesHtml = null, tableData = null, chartConfig = null) {
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
    
    // Add unique ID for normal messages
    const messageId = `msg-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;
    messageDiv.id = messageId;
    
    const icon = type === 'user' ? '👤' : type === 'error' ? '⚠️' : '🤖';
    const header = type === 'user' ? 'You' : type === 'error' ? 'Error' : 'Assistant';

    // Parse Markdown using marked.js
    let contentHtml = '';
    // Check if marked is available (it should be via CDN)
    if (typeof marked !== 'undefined' && marked.parse) {
        // Configure marked to not sanitize if you trust the content, or use DOMPurify if needed
        // For this use case, we trust the LLM output generally, but be aware of XSS risks
        contentHtml = marked.parse(content);
    } else {
        // Fallback if marked failed to load
        console.warn('Marked.js not found, falling back to simple formatting');
        contentHtml = formatContent(content); 
    }
    
    if (sqlQueriesHtml) {
        contentHtml += sqlQueriesHtml;
    }
    
    // Add chart rendering if available
    let chartHtml = '';
    if (chartConfig) {
        const chartId = `chart-${messageId}`;
        // Increased margin-bottom from 20px to 40px for better separation
        chartHtml = `<div class="chart-container-wrapper" style="position: relative; height: 300px; width: 100%; margin: 20px 0 40px 0;">
                        <canvas id="${chartId}"></canvas>
                     </div>`;
        // Store chart config to render after DOM update
        setTimeout(() => renderChart(chartId, chartConfig), 100);
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
        <div class="message-content">${contentHtml}${chartHtml}${tableHtml}</div>
    `;
    
    chatContainer.appendChild(messageDiv);
    
    // Render LaTeX math using KaTeX
    if (window.renderMathInElement) {
        renderMathInElement(messageDiv.querySelector('.message-content'), {
            delimiters: [
                {left: '$$', right: '$$', display: true},
                {left: '$', right: '$', display: false},
                {left: '\\(', right: '\\)', display: false},
                {left: '\\[', right: '\\]', display: true}
            ],
            throwOnError: false
        });
    }

    scrollToBottom();
    
    return messageDiv.id;
}

function safeStringifyChartConfig(config) {
    try {
        return JSON.stringify(config);
    } catch (err) {
        console.warn('Failed to serialize chart config for export:', err);
        return null;
    }
}

function renderChart(canvasId, config) {
    try {
        const canvas = document.getElementById(canvasId);
        if (!canvas) {
            console.error(`Canvas ${canvasId} not found for chart rendering.`);
            return;
        }
        const ctx = canvas.getContext('2d');
        // Ensure the config is valid and safe
        if (!config || !config.type || !config.data) {
            console.error("Invalid chart config:", config);
            return;
        }
        const serializedConfig = safeStringifyChartConfig(config);
        if (serializedConfig) {
            canvas.dataset.chartConfig = serializedConfig;
        }
        // Basic responsiveness
        config.options = config.options || {};
        config.options.responsive = true;
        config.options.maintainAspectRatio = false;
        
        // --- THEME CUSTOMIZATION FOR DARK MODE ---
        // Set default font color to white/light grey for visibility
        Chart.defaults.color = '#e0e0e0';
        Chart.defaults.borderColor = '#404040'; // Grid lines
        
        // Ensure plugins exist
        config.options.plugins = config.options.plugins || {};
        
        // Legend text color
        if (!config.options.plugins.legend) config.options.plugins.legend = {};
        if (!config.options.plugins.legend.labels) config.options.plugins.legend.labels = {};
        config.options.plugins.legend.labels.color = '#ffffff';
        
        // Title text color
        if (config.options.plugins.title) {
            config.options.plugins.title.color = '#ffffff';
            config.options.plugins.title.font = { size: 16, weight: 'bold' };
        }
        
        // Axis customization (Scales)
        config.options.scales = config.options.scales || {};
        const axes = ['x', 'y'];
        axes.forEach(axis => {
            if (!config.options.scales[axis]) config.options.scales[axis] = {};
            
            // Ticks (labels on axis)
            if (!config.options.scales[axis].ticks) config.options.scales[axis].ticks = {};
            config.options.scales[axis].ticks.color = '#e0e0e0';
            
            // Grid lines
            if (!config.options.scales[axis].grid) config.options.scales[axis].grid = {};
            config.options.scales[axis].grid.color = 'rgba(255, 255, 255, 0.1)';
            
            // Axis Titles
            if (config.options.scales[axis].title) {
                config.options.scales[axis].title.color = '#ffffff';
            }
        });
        
        // Brighten up the dataset colors if they are default
        if (config.data.datasets) {
            config.data.datasets.forEach(dataset => {
                // If it looks like the default muted teal, replace with a brighter cyan/blue
                if (dataset.backgroundColor === "rgba(75, 192, 192, 0.2)" || dataset.backgroundColor === "rgba(75, 192, 192, 0.6)") {
                    dataset.backgroundColor = "rgba(56, 189, 248, 0.6)"; // Sky blue
                    dataset.borderColor = "rgba(56, 189, 248, 1)";
                }
            });
        }
        // -----------------------------------------
        
        new Chart(ctx, config);
    } catch (e) {
        console.error("Error rendering chart:", e);
    }
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

function hasMessagesForCurrentSession() {
    if (!currentSessionId) {
        return false;
    }
    const messages = chatMessages[currentSessionId];
    return Array.isArray(messages) && messages.length > 0;
}

function updateDownloadButtonState() {
    if (!downloadPdfButton) return;
    downloadPdfButton.disabled = !hasMessagesForCurrentSession();
}

function cloneMessageNode(node) {
    const cloned = node.cloneNode(true);
    const imagePromises = [];

    // Convert original canvases (Chart.js) to inline PNGs in the cloned node
    const originalCanvases = node.querySelectorAll('canvas');
    const clonedCanvases = cloned.querySelectorAll('canvas');
    originalCanvases.forEach((originalCanvas, idx) => {
        const targetCanvas = clonedCanvases[idx];
        if (!targetCanvas) return;
        try {
            const img = document.createElement('img');
            img.src = getPrintReadyChartDataURL(originalCanvas);
            img.style.maxWidth = '100%';
            img.style.display = 'block';
            img.style.margin = '1rem 0';
            targetCanvas.replaceWith(img);

            if (img.decode) {
                imagePromises.push(img.decode().catch(() => {}));
            } else {
                imagePromises.push(waitForImageLoad(img));
            }
        } catch (err) {
            console.warn('Failed to convert chart canvas for PDF export:', err);
        }
    });

    // Ensure existing images inside the cloned node are loaded
    const images = cloned.querySelectorAll('img');
    images.forEach(img => {
        if (img.decode) {
            imagePromises.push(img.decode().catch(() => {}));
        } else {
            imagePromises.push(waitForImageLoad(img));
        }
    });

    return { cloned, imagePromises };
}

function buildPrintableChatDocument() {
    const host = document.createElement('div');
    host.style.position = 'fixed';
    host.style.left = '-9999px';
    host.style.top = '0';
    host.style.width = '900px';
    host.style.backgroundColor = '#ffffff';
    host.style.color = '#111827';
    host.style.fontFamily = "Inter, 'Helvetica Neue', Arial, sans-serif";

    const wrapper = document.createElement('div');
    wrapper.style.padding = '24px';
    wrapper.style.lineHeight = '1.5';
    wrapper.style.fontSize = '0.95rem';
    wrapper.style.backgroundColor = '#ffffff';

    const title = document.createElement('h1');
    title.textContent = 'Database Query Assistant • Chat Transcript';
    title.style.fontSize = '1.5rem';
    title.style.marginBottom = '0.5rem';
    title.style.color = '#0f172a';
    wrapper.appendChild(title);

    const meta = document.createElement('p');
    meta.style.margin = '0 0 1.5rem 0';
    meta.style.color = '#475569';
    meta.textContent = `Exported: ${new Date().toLocaleString()}`;
    wrapper.appendChild(meta);

    const assetPromises = [];

    const messageNodes = chatContainer.querySelectorAll('.message');
    if (messageNodes.length === 0) {
        const empty = document.createElement('p');
        empty.textContent = 'No chat messages to export yet.';
        empty.style.fontStyle = 'italic';
        wrapper.appendChild(empty);
    } else {
        messageNodes.forEach(node => {
            const { cloned, imagePromises } = cloneMessageNode(node);
            assetPromises.push(...imagePromises);

            ['message', 'user', 'bot', 'error'].forEach(cls => cloned.classList.remove(cls));
            cloned.style.backgroundColor = '#ffffff';
            cloned.style.color = '#111827';
            cloned.style.border = '1px solid #e2e8f0';
            cloned.style.boxShadow = 'none';
            cloned.style.borderRadius = '10px';
            cloned.style.padding = '16px 20px';
            cloned.style.marginBottom = '16px';

            const header = cloned.querySelector('.message-header');
            if (header) {
                header.classList.remove('message-header');
                header.style.display = 'flex';
                header.style.alignItems = 'center';
                header.style.marginBottom = '8px';
                header.style.fontWeight = '600';
                header.style.fontSize = '0.95rem';
                header.style.color = '#0f172a';

                const icon = header.querySelector('.icon');
                if (icon) {
                    icon.style.marginRight = '8px';
                    icon.style.color = '#2563eb';
                }
            }

            const content = cloned.querySelector('.message-content');
            if (content) {
                content.style.color = '#1f2937';
                content.style.whiteSpace = 'normal';
                content.style.backgroundColor = 'transparent';
                content.style.lineHeight = '1.6';

                content.querySelectorAll('code').forEach(code => {
                    code.style.backgroundColor = '#f1f5f9';
                    code.style.color = '#0f172a';
                    code.style.padding = '2px 4px';
                    code.style.borderRadius = '4px';
                });

                content.querySelectorAll('pre').forEach(pre => {
                    pre.style.backgroundColor = '#f1f5f9';
                    pre.style.color = '#0f172a';
                    pre.style.padding = '10px 12px';
                    pre.style.borderRadius = '6px';
                });

                content.querySelectorAll('table').forEach(table => {
                    table.style.backgroundColor = '#ffffff';
                    table.style.color = '#0f172a';
                    table.style.border = '1px solid #e2e8f0';
                    table.style.borderCollapse = 'collapse';
                    table.style.width = '100%';
                    table.querySelectorAll('th, td').forEach(cell => {
                        cell.style.border = '1px solid #e2e8f0';
                        cell.style.padding = '6px 8px';
                        cell.style.backgroundColor = '#ffffff';
                        cell.style.color = '#0f172a';
                        cell.style.fontWeight = '400';
                    });
                    table.querySelectorAll('th').forEach(headerCell => {
                        headerCell.style.fontWeight = '600';
                        headerCell.style.backgroundColor = '#f8fafc';
                        headerCell.style.color = '#0f172a';
                    });
                });

                const tablesInfo = cloned.querySelectorAll('.table-info-message');
                tablesInfo.forEach(info => {
                    info.style.backgroundColor = '#f8fafc';
                    info.style.color = '#0f172a';
                    info.style.border = '1px dashed #cbd5f5';
                    info.style.padding = '8px 10px';
                });

                const sqlBlocks = cloned.querySelectorAll('.sql-query');
                sqlBlocks.forEach(sql => {
                    sql.style.backgroundColor = '#f1f5f9';
                    sql.style.color = '#0f172a';
                    sql.style.borderRadius = '6px';
                    sql.style.padding = '10px 12px';
                });

                const sqlTitle = cloned.querySelector('.sql-queries-title');
                if (sqlTitle) {
                    sqlTitle.style.color = '#0f172a';
                }

                const sqlContainer = cloned.querySelector('.sql-queries');
                if (sqlContainer) {
                    sqlContainer.style.borderTop = '1px solid #e2e8f0';
                }

                const tableActionBars = cloned.querySelectorAll('.table-actions');
                tableActionBars.forEach(actions => actions.remove());

            }

            wrapper.appendChild(cloned);
        });
    }

    host.appendChild(wrapper);
    document.body.appendChild(host);
    return { host, wrapper, assetPromises };
}

function waitForImageLoad(img) {
    return new Promise(resolve => {
        if (!img) {
            resolve();
            return;
        }
        if (img.complete) {
            resolve();
        } else {
            const cleanup = () => {
                img.onload = null;
                img.onerror = null;
                resolve();
            };
            img.onload = cleanup;
            img.onerror = cleanup;
        }
    });
}

function applyPrintThemeToChartConfig(printConfig) {
    printConfig.options = printConfig.options || {};
    printConfig.options.animation = false;
    printConfig.options.responsive = false;
    printConfig.options.maintainAspectRatio = false;

    printConfig.options.plugins = printConfig.options.plugins || {};
    const plugins = printConfig.options.plugins;
    plugins.legend = plugins.legend || {};
    plugins.legend.labels = plugins.legend.labels || {};
    plugins.legend.labels.color = '#111827';
    plugins.legend.labels.font = plugins.legend.labels.font || {};
    plugins.legend.labels.font.size = 20;
    plugins.legend.labels.font.family = "'Inter', 'Helvetica Neue', Arial, sans-serif";

    if (plugins.title) {
        plugins.title.color = '#111827';
        plugins.title.font = plugins.title.font || {};
        plugins.title.font.size = 28;
        plugins.title.font.family = "'Inter', 'Helvetica Neue', Arial, sans-serif";
        plugins.title.font.weight = '600';
    }

    printConfig.options.scales = printConfig.options.scales || {};
    ['x', 'y'].forEach(axis => {
        const scale = printConfig.options.scales[axis] || {};
        scale.ticks = scale.ticks || {};
        scale.ticks.color = '#111827';
        scale.ticks.font = scale.ticks.font || {};
        scale.ticks.font.size = 13;
        scale.ticks.font.family = "'Inter', 'Helvetica Neue', Arial, sans-serif";
        scale.grid = scale.grid || {};
        scale.grid.color = 'rgba(15, 23, 42, 0.15)';
        scale.title = scale.title || {};
        scale.title.color = '#111827';
        scale.title.font = scale.title.font || {};
        scale.title.font.size = 16;
        scale.title.font.family = "'Inter', 'Helvetica Neue', Arial, sans-serif";
        scale.title.font.weight = '600';
        printConfig.options.scales[axis] = scale;
    });

    const backgroundPlugin = {
        id: 'printBackground',
        beforeDraw(chart, args, opts) {
            const ctx = chart.canvas.getContext('2d');
            ctx.save();
            ctx.fillStyle = (opts && opts.color) || '#ffffff';
            ctx.fillRect(0, 0, chart.width, chart.height);
            ctx.restore();
        }
    };

    printConfig.plugins = printConfig.plugins || [];
    printConfig.plugins.push(backgroundPlugin);
    plugins.printBackground = { color: '#ffffff' };

    if (printConfig.data && Array.isArray(printConfig.data.datasets)) {
        printConfig.data.datasets.forEach(dataset => {
            if (!dataset.borderColor) {
                dataset.borderColor = '#2563eb';
            }
            if (!dataset.backgroundColor) {
                dataset.backgroundColor = 'rgba(37, 99, 235, 0.25)';
            }
        });
    }
}

function getPrintReadyChartDataURL(originalCanvas) {
    if (typeof Chart === 'undefined') {
        return originalCanvas.toDataURL('image/png');
    }
    const configJson = originalCanvas.dataset.chartConfig;
    if (!configJson) {
        return originalCanvas.toDataURL('image/png');
    }
    try {
        const printConfig = JSON.parse(configJson);
        applyPrintThemeToChartConfig(printConfig);
        const tempCanvas = document.createElement('canvas');
        tempCanvas.width = originalCanvas.width;
        tempCanvas.height = originalCanvas.height;
        const ctx = tempCanvas.getContext('2d');
        const tempChart = new Chart(ctx, printConfig);
        const dataUrl = tempCanvas.toDataURL('image/png');
        tempChart.destroy();
        return dataUrl;
    } catch (err) {
        console.warn('Failed to rebuild chart for PDF export, falling back to original canvas image:', err);
        return originalCanvas.toDataURL('image/png');
    }
}

function getChatExportFilename(messages) {
    const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
    const firstUserMessage = messages.find(msg => msg.type === 'user');
    let base = firstUserMessage ? firstUserMessage.content.slice(0, 40) : 'chat-transcript';
    base = base || 'chat-transcript';
    base = base.replace(/[^a-z0-9]+/gi, '-').replace(/^-+|-+$/g, '').toLowerCase();
    return `${base || 'chat-transcript'}-${timestamp}.pdf`;
}

function downloadCurrentChatAsPDF() {
    if (!downloadPdfButton || downloadPdfButton.disabled) {
        return;
    }
    if (typeof html2pdf === 'undefined') {
        console.error('html2pdf.js is not loaded, cannot export chat.');
        alert('PDF export library failed to load. Please refresh the page and try again.');
        return;
    }
    if (!hasMessagesForCurrentSession()) {
        console.warn('No chat messages available to export.');
        alert('Send a message in the current chat before exporting it to PDF.');
        return;
    }

    const messages = chatMessages[currentSessionId] || [];
    const { host, wrapper, assetPromises } = buildPrintableChatDocument();
    const filename = getChatExportFilename(messages);

    downloadPdfButton.disabled = true;

    const pdfOptions = {
        margin: 0.4,
        filename,
        image: { type: 'jpeg', quality: 0.95 },
        html2canvas: { scale: 1.5, useCORS: true, backgroundColor: '#ffffff', logging: false },
        jsPDF: { unit: 'in', format: 'letter', orientation: 'portrait' }
    };

    Promise.all(assetPromises)
        .catch(err => {
            console.warn('Some assets failed to load for PDF export:', err);
        })
        .finally(() => {
            html2pdf().set(pdfOptions).from(wrapper).save()
                .catch(error => {
                    console.error('Failed to export chat as PDF:', error);
                })
                .finally(() => {
                    if (host && host.parentNode) {
                        host.parentNode.removeChild(host);
                    }
                    updateDownloadButtonState();
                });
        });
}

// Focus input on load
window.addEventListener('load', () => {
    queryInput.focus();
});


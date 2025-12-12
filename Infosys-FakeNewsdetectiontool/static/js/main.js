/* Basic JavaScript for TruthGuard */

// Toggle password visibility
function togglePasswordVisibility(inputId) {
    const input = document.getElementById(inputId);
    if (input.type === 'password') {
        input.type = 'text';
    } else {
        input.type = 'password';
    }
}

// Show/hide loading spinner
function showLoading(show = true) {
    const loadingEl = document.getElementById('loading');
    if (loadingEl) {
        loadingEl.style.display = show ? 'block' : 'none';
    }
}

// Copy text to clipboard
function copyToClipboard(text) {
    navigator.clipboard.writeText(text).then(() => {
        alert('Copied to clipboard!');
    });
}

// Format date
function formatDate(dateString) {
    const date = new Date(dateString);
    return date.toLocaleDateString() + ' ' + date.toLocaleTimeString();
}
function displayResults(data) {
    const resultsDiv = document.getElementById('analysisResults');
    if (!resultsDiv) return;

    if (data.short_report) {
        // Show compact text block
        let html = `
            <div class="result-card result-${data.compact ? data.compact.classification?.toLowerCase() : 'summary'}">
                <h3>Quick Summary</h3>
                <pre style="white-space:pre-wrap; font-family:inherit;">${data.short_report}</pre>
                <p><a href="/analysis/${data.analysis_id}">View Detailed Analysis</a></p>
            </div>
        `;
        resultsDiv.innerHTML = html;
        resultsDiv.style.display = 'block';
        return;
    }

    // fallback to existing UI
    let html = `
        <div class="result-card result-${data.classification.toLowerCase()}">
            <h3>Analysis Results</h3>
            <p><strong>Classification:</strong> ${data.classification}</p>
            <p><strong>Confidence:</strong> ${data.confidence}%</p>
            <p><strong>Processing Time:</strong> ${data.processing_ms}ms</p>
            <p><strong>Word Count:</strong> ${data.word_count}</p>
            <p><a href="/analysis/${data.analysis_id}">View Detailed Analysis</a></p>
        </div>
    `;
    resultsDiv.innerHTML = html;
    resultsDiv.style.display = 'block';
}

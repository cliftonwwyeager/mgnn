document.addEventListener('DOMContentLoaded', function() {
    const uploadForm = document.getElementById('upload-form');
    const uploadBox = document.getElementById('upload-box');
    const uploadBtn = document.getElementById('upload-btn');
    const confirmForm = document.getElementById('confirm-form');
    const progressBar = document.getElementById('progress-bar');
    const progress = document.getElementById('progress');
    const progressText = document.getElementById('progress-text');

    function updateProgress(percentage, timeRemaining) {
        progress.style.width = percentage + '%';
        progressText.textContent = `Processed: ${percentage}%, Estimated Time Remaining: ${timeRemaining}s`;
    }

    if (uploadBox) {
        uploadBox.addEventListener('click', () => {
            document.getElementById('file-input').click();
        });

        uploadBox.addEventListener('dragover', (e) => {
            e.preventDefault();
            uploadBox.classList.add('dragover');
        });

        uploadBox.addEventListener('dragleave', () => {
            uploadBox.classList.remove('dragover');
        });

        uploadBox.addEventListener('drop', (e) => {
            e.preventDefault();
            uploadBox.classList.remove('dragover');
            const files = e.dataTransfer.files;
            document.getElementById('file-input').files = files;
            uploadForm.submit();
        });
    }

    if (uploadBtn) {
        uploadBtn.addEventListener('click', () => {
            document.getElementById('file-input').click();
        });

        uploadForm.addEventListener('submit', function(event) {
            event.preventDefault();
            const formData = new FormData(uploadForm);
            fetch('/upload', {
                method: 'POST',
                body: formData
            })
            .then(response => response.json())
            .then(data => {
                const resultDiv = document.getElementById('upload-result');
                resultDiv.innerHTML = `<p>Prediction: ${data.predicted_class}, Confidence: ${data.confidence}</p>`;
            })
            .catch(error => {
                console.error('Error:', error);
            });
        });
    }

    if (confirmForm) {
        confirmForm.addEventListener('submit', function(event) {
            event.preventDefault();
            const formData = new FormData(confirmForm);
            fetch('/confirm', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    hash: formData.get('hash'),
                    filename: formData.get('filename'),
                    is_correct: formData.get('is_correct') === 'true'
                })
            })
            .then(response => response.json())
            .then(data => {
                const resultDiv = document.getElementById('confirm-result');
                resultDiv.innerHTML = `<p>Status: ${data.status}</p>`;
            })
            .catch(error => {
                console.error('Error:', error);
            });
        });
    }

    if (window.location.pathname === '/results') {
        fetch('/get_results')
        .then(response => response.json())
        .then(data => {
            const resultsDiv = document.getElementById('results');
            if (data.error) {
                resultsDiv.innerHTML = `<p>${data.error}</p>`;
            } else {
                const table = document.createElement('table');
                const headers = ['Filename', 'Hash', 'Predicted Class', 'Confidence', 'Is Correct', 'Timestamp'];
                const headerRow = document.createElement('tr');
                headers.forEach(header => {
                    const th = document.createElement('th');
                    th.textContent = header;
                    headerRow.appendChild(th);
                });
                table.appendChild(headerRow);

                data.forEach(row => {
                    const tr = document.createElement('tr');
                    headers.forEach(header => {
                        const td = document.createElement('td');
                        td.textContent = row[header.toLowerCase()];
                        tr.appendChild(td);
                    });
                    table.appendChild(tr);
                });

                resultsDiv.appendChild(table);
            }
        })
        .catch(error => {
            console.error('Error:', error);
        });
    }
});

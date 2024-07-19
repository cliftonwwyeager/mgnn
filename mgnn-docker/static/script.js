document.addEventListener('DOMContentLoaded', function() {
    const uploadForm = document.getElementById('upload-form');
    const uploadBox = document.getElementById('upload-box');
    const uploadBtn = document.getElementById('upload-btn');
    const confirmForm = document.getElementById('confirm-form');

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
                resultDiv.innerHTML = `<p>Prediction: ${data.prediction}</p>`;
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
                    hash: formData.get('hash')
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
});

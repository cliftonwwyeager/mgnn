function uploadFile() {
    var formData = new FormData();
    var fileInput = document.getElementById('fileInput');
    var file = fileInput.files[0];
    formData.append('file', file);

    fetch('/upload', {
        method: 'POST',
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        document.getElementById('result').innerText = `Predicted Class: ${data.predicted_class}, Confidence: ${data.confidence}`;
    })
    .catch(error => {
        console.error('Error:', error);
    });
}

document.addEventListener('DOMContentLoaded', function () {
    const uploadBox = document.getElementById('upload-box');
    const uploadForm = document.getElementById('upload-form');
    const fileInput = document.getElementById('file-input');
    const uploadBtn = document.getElementById('upload-btn');
    const uploadResult = document.getElementById('upload-result');
    const progressBar = document.getElementById('progress-bar');
    const progress = document.getElementById('progress');
    const progressText = document.getElementById('progress-text');
    const confirmForm = document.getElementById('confirm-form');
    const confirmResult = document.getElementById('confirm-result');
    
    if (uploadBox) {
        uploadBox.addEventListener('click', () => fileInput.click());
        uploadBox.addEventListener('dragover', (e) => {
            e.preventDefault();
            uploadBox.classList.add('dragover');
        });
        uploadBox.addEventListener('dragleave', () => uploadBox.classList.remove('dragover'));
        uploadBox.addEventListener('drop', (e) => {
            e.preventDefault();
            uploadBox.classList.remove('dragover');
            fileInput.files = e.dataTransfer.files;
            uploadForm.style.display = 'block';
        });
    }

    if (uploadForm) {
        uploadForm.addEventListener('submit', (e) => {
            e.preventDefault();
            const formData = new FormData(uploadForm);
            const xhr = new XMLHttpRequest();
            xhr.open('POST', '/upload', true);
            
            xhr.upload.onprogress = function (e) {
                if (e.lengthComputable) {
                    const percentComplete = (e.loaded / e.total) * 100;
                    progressBar.style.display = 'block';
                    progress.style.width = percentComplete + '%';
                    progressText.textContent = percentComplete.toFixed(2) + '%';
                }
            };

            xhr.onload = function () {
                if (xhr.status === 200) {
                    uploadResult.textContent = 'Upload successful!';
                    uploadResult.style.color = '#00FF00';
                } else {
                    uploadResult.textContent = 'Upload failed!';
                    uploadResult.style.color = 'red';
                }
                progressBar.style.display = 'none';
            };

            xhr.send(formData);
        });
    }

    if (uploadBtn) {
        uploadBtn.addEventListener('click', () => fileInput.click());
    }

    if (confirmForm) {
        confirmForm.addEventListener('submit', (e) => {
            e.preventDefault();
            const formData = new FormData(confirmForm);
            fetch('/confirm', {
                method: 'POST',
                body: JSON.stringify({
                    hash: formData.get('hash'),
                    filename: formData.get('filename'),
                    is_correct: formData.get('is_correct')
                }),
                headers: {
                    'Content-Type': 'application/json'
                }
            })
            .then(response => response.json())
            .then(data => {
                if (data.status === 'Confirmation recorded') {
                    confirmResult.textContent = 'Confirmation recorded successfully!';
                    confirmResult.style.color = '#00FF00';
                } else {
                    confirmResult.textContent = 'Confirmation failed!';
                    confirmResult.style.color = 'red';
                }
            })
            .catch(error => {
                confirmResult.textContent = 'An error occurred!';
                confirmResult.style.color = 'red';
            });
        });
    }
});

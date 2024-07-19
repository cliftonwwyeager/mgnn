const uploadBox = document.getElementById('upload-box');
const processBtn = document.getElementById('process-btn');
const resultsDiv = document.getElementById('results');

uploadBox.addEventListener('dragover', (event) => {
    event.preventDefault();
    uploadBox.classList.add('dragover');
});

uploadBox.addEventListener('dragleave', () => {
    uploadBox.classList.remove('dragover');
});

uploadBox.addEventListener('drop', (event) => {
    event.preventDefault();
    uploadBox.classList.remove('dragover');
    handleFiles(event.dataTransfer.files);
});

uploadBox.addEventListener('click', () => {
    const fileInput = document.createElement('input');
    fileInput.type = 'file';
    fileInput.multiple = true;
    fileInput.onchange = () => handleFiles(fileInput.files);
    fileInput.click();
});

function handleFiles(files) {
    for (let file of files) {
        uploadFile(file);
    }
}

function uploadFile(file) {
    const formData = new FormData();
    formData.append('file', file);
    fetch('/api/upload', {
        method: 'POST',
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        if (data.error) {
            alert(data.error);
        } else {
            alert(data.message);
        }
    });
}

processBtn.addEventListener('click', () => {
    fetch('/api/process', {
        method: 'GET'
    })
    .then(response => response.json())
    .then(data => {
        alert(data.message);
        fetchResults();
    });
});

function fetchResults() {
    fetch('/api/results')
    .then(response => response.json())
    .then(data => {
        resultsDiv.innerHTML = '';
        for (let file of data) {
            const div = document.createElement('div');
            div.innerHTML = `File: ${file.file_hash} - Malware Detected: ${file.result.malware_detected}`;
            const yesNoSelect = document.createElement('select');
            yesNoSelect.innerHTML = '<option value="yes">Yes</option><option value="no">No</option>';
            yesNoSelect.onchange = () => submitFeedback(file.file_hash, yesNoSelect.value);
            div.appendChild(yesNoSelect);
            resultsDiv.appendChild(div);
        }
    });
}

function submitFeedback(fileHash, feedback) {
    fetch('/api/feedback', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({ file_hash: fileHash, feedback: feedback })
    })
    .then(response => response.json())
    .then(data => {
        alert(data.message);
    });
}
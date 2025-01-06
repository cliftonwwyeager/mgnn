document.addEventListener('DOMContentLoaded', function () {
    const uploadBox = document.getElementById('upload-box');
    const fileInput = document.getElementById('file-input');
    const uploadResult = document.getElementById('upload-result');
    const resultContainer = document.getElementById('result-container');
    const pathInput = document.getElementById('path-input');
    const recursiveCheck = document.getElementById('recursive-check');
    const scanButton = document.getElementById('scan-button');

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
            handleFiles(e.dataTransfer.files);
        });
    }

    if (fileInput) {
        fileInput.addEventListener('change', (e) => {
            handleFiles(e.target.files);
        });
    }

    if (scanButton) {
        scanButton.addEventListener('click', handleScan);
    }

    function handleFiles(files) {
        const results = [];
        let uploadedCount = 0;
        uploadResult.textContent = 'Uploading...';

        for (let i = 0; i < files.length; i++) {
            const formData = new FormData();
            formData.append('file', files[i]);

            fetch('/upload', {
                method: 'POST',
                body: formData
            })
            .then(response => response.json())
            .then(data => {
                const itemResult = {
                    fileName: files[i].name,
                    predictedClass: data.predicted_class,
                    confidence: data.confidence,
                    isMalware: data.predicted_class === 1,
                    executionSteps: []
                };
                results.push(itemResult);
                uploadedCount++;
                if (uploadedCount === files.length) {
                    uploadResult.textContent = 'Processing complete';
                    displayResults(results);
                }
            })
            .catch(error => {
                console.error('Error:', error);
                results.push({
                    fileName: files[i].name,
                    predictedClass: 'N/A',
                    confidence: 0,
                    isMalware: false,
                    executionSteps: [],
                    error: error.toString()
                });
                uploadedCount++;
                if (uploadedCount === files.length) {
                    uploadResult.textContent = 'Error during upload';
                    displayResults(results);
                }
            });
        }
    }

    function handleScan() {
        const pathValue = pathInput.value.trim();
        if (!pathValue) {
            alert('Please enter a directory or Samba path.');
            return;
        }
        const isRecursive = recursiveCheck.checked;
        let username = '';
        let password = '';

        if (pathValue.toLowerCase().startsWith('smb://')) {
            username = prompt('Samba Username (leave blank if none):') || '';
            password = prompt('Samba Password (leave blank if none):') || '';
        }

        uploadResult.textContent = 'Scanning in progress...';
        const bodyData = {
            path: pathValue,
            recursive: isRecursive,
            smbUser: username,
            smbPass: password
        };

        fetch('/scan', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(bodyData)
        })
        .then(response => response.json())
        .then(data => {
            uploadResult.textContent = 'Scan complete';
            displayResults(data);
        })
        .catch(error => {
            console.error('Scan error:', error);
            uploadResult.textContent = 'Error during scan';
        });
    }

    function displayResults(data) {
        resultContainer.innerHTML = '';
        data.forEach(fileResult => {
            const resultItem = document.createElement('div');
            resultItem.classList.add('result-item');

            const fileName = document.createElement('p');
            fileName.textContent = `File/Path: ${fileResult.fileName || fileResult.path || 'N/A'}`;

            const predictedClass = document.createElement('p');
            predictedClass.textContent = `Predicted Class: ${fileResult.predictedClass || 'N/A'}`;

            const confidence = document.createElement('p');
            confidence.textContent = `Confidence: ${parseFloat(fileResult.confidence || 0).toFixed(2)}`;

            resultItem.appendChild(fileName);
            resultItem.appendChild(predictedClass);
            resultItem.appendChild(confidence);

            if (fileResult.isMalware) {
                const timelineTitle = document.createElement('p');
                timelineTitle.textContent = 'Execution Steps:';
                resultItem.appendChild(timelineTitle);

                const timeline = document.createElement('div');
                timeline.classList.add('timeline');
                (fileResult.executionSteps || []).forEach(step => {
                    const stepItem = document.createElement('p');
                    stepItem.textContent = `${step.stepNumber}: ${step.description}`;
                    timeline.appendChild(stepItem);
                });
                resultItem.appendChild(timeline);
            }

            if (fileResult.error) {
                const errorEl = document.createElement('p');
                errorEl.style.color = 'red';
                errorEl.textContent = `Error: ${fileResult.error}`;
                resultItem.appendChild(errorEl);
            }

            resultContainer.appendChild(resultItem);
        });
    }
});

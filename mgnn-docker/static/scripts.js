document.addEventListener('DOMContentLoaded', function () {
    const uploadBox = document.getElementById('upload-box');
    const fileInput = document.getElementById('file-input');
    const uploadResult = document.getElementById('upload-result');
    const resultContainer = document.getElementById('result-container');

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
                    error: null
                };
                results.push(itemResult);
            })
            .catch(error => {
                console.error('Error:', error);
                results.push({
                    fileName: files[i].name,
                    predictedClass: 'N/A',
                    confidence: 0,
                    isMalware: false,
                    error: error.toString()
                });
            })
            .finally(() => {
                uploadedCount++;
                if (uploadedCount === files.length) {
                    uploadResult.textContent = 'Processing complete';
                    displayResults(results);
                }
            });
        }
    }

    function displayResults(data) {
        resultContainer.innerHTML = '';
        data.forEach(fileResult => {
            const resultItem = document.createElement('div');
            resultItem.classList.add('result-item');

            const fileName = document.createElement('p');
            fileName.textContent = `File: ${fileResult.fileName || 'N/A'}`;
            resultItem.appendChild(fileName);

            const predictedClass = document.createElement('p');
            predictedClass.textContent = `Predicted Class: ${fileResult.predictedClass}`;
            resultItem.appendChild(predictedClass);

            const confidence = document.createElement('p');
            confidence.textContent = `Confidence: ${parseFloat(fileResult.confidence || 0).toFixed(2)}`;
            resultItem.appendChild(confidence);

            if (fileResult.isMalware) {
                const warning = document.createElement('p');
                warning.style.color = 'red';
                warning.textContent = 'Malware detected!';
                resultItem.appendChild(warning);
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

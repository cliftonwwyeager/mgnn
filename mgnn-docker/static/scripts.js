<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Malware Detection</title>
    <style>
        #upload-box {
            border: 2px dashed #cccccc;
            padding: 20px;
            text-align: center;
            cursor: pointer;
        }
        #upload-box.dragover {
            border-color: #0000FF;
        }
        #result-container {
            margin-top: 20px;
        }
        .result-item {
            margin-bottom: 20px;
            padding: 10px;
            border: 1px solid #cccccc;
        }
        .timeline {
            margin-top: 10px;
        }
    </style>
</head>
<body>
    <div id="upload-box">Drag and drop files here or click to upload</div>
    <input type="file" id="file-input" multiple style="display: none;">
    <div id="upload-result"></div>
    <div id="result-container"></div>

    <script>
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

            fileInput.addEventListener('change', (e) => {
                handleFiles(e.target.files);
            });

            function handleFiles(files) {
                const formData = new FormData();
                for (let i = 0; i < files.length; i++) {
                    formData.append('files[]', files[i]);
                }

                fetch('/upload', {
                    method: 'POST',
                    body: formData
                })
                .then(response => response.json())
                .then(data => {
                    uploadResult.textContent = 'Processing complete';
                    displayResults(data);
                })
                .catch(error => {
                    console.error('Error:', error);
                    uploadResult.textContent = 'Error occurred during file upload.';
                });
            }

            function displayResults(data) {
                resultContainer.innerHTML = ''; // Clear previous results
                data.forEach(fileResult => {
                    const resultItem = document.createElement('div');
                    resultItem.classList.add('result-item');

                    const fileName = document.createElement('p');
                    fileName.textContent = `File: ${fileResult.fileName}`;

                    const predictedClass = document.createElement('p');
                    predictedClass.textContent = `Predicted Class: ${fileResult.predictedClass}`;

                    const confidence = document.createElement('p');
                    confidence.textContent = `Confidence: ${fileResult.confidence.toFixed(2)}%`;

                    resultItem.appendChild(fileName);
                    resultItem.appendChild(predictedClass);
                    resultItem.appendChild(confidence);

                    if (fileResult.isMalware) {
                        const timelineTitle = document.createElement('p');
                        timelineTitle.textContent = 'Execution Steps:';
                        resultItem.appendChild(timelineTitle);

                        const timeline = document.createElement('div');
                        timeline.classList.add('timeline');
                        fileResult.executionSteps.forEach(step => {
                            const stepItem = document.createElement('p');
                            stepItem.textContent = `${step.stepNumber}: ${step.description}`;
                            timeline.appendChild(stepItem);
                        });
                        resultItem.appendChild(timeline);
                    }

                    resultContainer.appendChild(resultItem);
                });
            }
        });
    </script>
</body>
</html>

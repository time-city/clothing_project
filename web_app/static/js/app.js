// app.js - Frontend logic for clothing classifier (Multi-Model)

let selectedImage = null;
let chart = null;
let currentModel = 'resnet';

// DOM Elements
const uploadArea = document.getElementById('uploadArea');
const imageInput = document.getElementById('imageInput');
const uploadBtn = document.getElementById('uploadBtn');
const previewSection = document.getElementById('previewSection');
const previewImage = document.getElementById('previewImage');
const fileName = document.getElementById('fileName');
const fileSize = document.getElementById('fileSize');
const buttonGroup = document.getElementById('buttonGroup');
const predictBtn = document.getElementById('predictBtn');
const resetBtn = document.getElementById('resetBtn');
const spinner = document.getElementById('spinner');
const resultSection = document.getElementById('resultSection');
const errorMessage = document.getElementById('errorMessage');
const deviceInfo = document.getElementById('deviceInfo');
const modelInfo = document.getElementById('modelInfo');
const modelTypeSelect = document.getElementById('modelType');

// ==================== Initialize ====================
document.addEventListener('DOMContentLoaded', function() {
    setupEventListeners();
    checkStatus();
    loadModelInfo();
});

function setupEventListeners() {
    uploadBtn.addEventListener('click', () => imageInput.click());
    imageInput.addEventListener('change', handleFileSelect);
    uploadArea.addEventListener('dragover', handleDragOver);
    uploadArea.addEventListener('dragleave', handleDragLeave);
    uploadArea.addEventListener('drop', handleDrop);
    document.addEventListener('paste', handlePaste);
    predictBtn.addEventListener('click', predict);
    resetBtn.addEventListener('click', reset);
    modelTypeSelect.addEventListener('change', handleModelChange);
}

function handleModelChange(event) {
    currentModel = event.target.value;
    localStorage.setItem('selectedModel', currentModel);
    console.log(`Model switched to: ${currentModel}`);
    showError(`✅ Model đã chuyển sang: ${currentModel === 'resnet' ? 'ResNet50' : 'MobileNet'}`);
    setTimeout(() => errorMessage.style.display = 'none', 3000);
}

async function loadModelInfo() {
    try {
        const response = await fetch('/api/models');
        const data = await response.json();
        
        const resnetAvailable = data.available.resnet;
        const mobilenetAvailable = data.available.mobilenet;
        
        // Update select options
        modelTypeSelect.options[0].disabled = !resnetAvailable;
        modelTypeSelect.options[1].disabled = !mobilenetAvailable;
        
        // Restore saved model preference or use current
        const saved = localStorage.getItem('selectedModel');
        if (saved && document.querySelector(`option[value="${saved}"]`).disabled === false) {
            currentModel = saved;
            modelTypeSelect.value = currentModel;
        } else {
            currentModel = data.current.type;
            modelTypeSelect.value = currentModel;
        }
        
        // Update model info
        let info = `📦 Model: ${data.current.variant}`;
        if (resnetAvailable) info += ' ✅ ResNet50';
        if (!resnetAvailable) info += ' ❌ ResNet50';
        if (mobilenetAvailable) info += ' ✅ MobileNet';
        if (!mobilenetAvailable) info += ' ❌ MobileNet';

        if (Array.isArray(data.current.classes) && data.current.classes.length > 0) {
            info += ` | Lớp: ${data.current.classes.join(', ')}`;
        }
        
        modelInfo.textContent = info;
    } catch (error) {
        console.error('Error loading model info:', error);
        modelInfo.textContent = '⚠️ Không thể tải thông tin model';
    }
}

// ==================== File Handling ====================
function handleFileSelect(event) {
    const file = event.target.files[0];
    if (file) {
        processFile(file);
    }
}

function handleDragOver(event) {
    event.preventDefault();
    event.stopPropagation();
    uploadArea.classList.add('dragover');
}

function handleDragLeave(event) {
    event.preventDefault();
    event.stopPropagation();
    uploadArea.classList.remove('dragover');
}

function handleDrop(event) {
    event.preventDefault();
    event.stopPropagation();
    uploadArea.classList.remove('dragover');
    
    const files = event.dataTransfer.files;
    if (files.length > 0) {
        const file = files[0];
        if (file.type.startsWith('image/')) {
            processFile(file);
        } else {
            showError('Vui lòng chọn tệp ảnh hợp lệ!');
        }
    }
}

function handlePaste(event) {
    const clipboardItems = event.clipboardData?.items;
    if (!clipboardItems) {
        return;
    }

    for (const item of clipboardItems) {
        if (item.type && item.type.startsWith('image/')) {
            const file = item.getAsFile();
            if (file) {
                processFile(file);
                return;
            }
        }
    }
}

function processFile(file) {
    // Check file size
    if (file.size > 16 * 1024 * 1024) {
        showError('Kích thước tệp quá lớn (tối đa 16MB)!');
        return;
    }
    
    // Store file
    selectedImage = file;
    
    // Show preview
    const reader = new FileReader();
    reader.onload = function(e) {
        previewImage.src = e.target.result;
        fileName.textContent = `📄 ${file.name}`;
        fileSize.textContent = `📊 ${formatFileSize(file.size)}`;
        
        previewSection.style.display = 'block';
        buttonGroup.style.display = 'flex';
        errorMessage.style.display = 'none';
        resultSection.style.display = 'none';
    };
    reader.readAsDataURL(file);
}

function formatFileSize(bytes) {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return Math.round(bytes / Math.pow(k, i) * 100) / 100 + ' ' + sizes[i];
}

// ==================== Prediction ====================
async function predict() {
    if (!selectedImage) {
        showError('Vui lòng chọn ảnh!');
        return;
    }
    
    // Show spinner, hide results
    spinner.style.display = 'block';
    resultSection.style.display = 'none';
    errorMessage.style.display = 'none';
    predictBtn.disabled = true;
    
    try {
        // Create FormData
        const formData = new FormData();
        formData.append('image', selectedImage);
        
        // Send request
        const response = await fetch('/api/predict', {
            method: 'POST',
            body: formData
        });
        
        const data = await response.json();
        
        if (!response.ok) {
            showError(data.error || 'Dự đoán thất bại!');
            spinner.style.display = 'none';
            predictBtn.disabled = false;
            return;
        }
        
        if (data.success) {
            displayResults(data);
        } else {
            showError(data.error || 'Lỗi dự đoán!');
        }
    } catch (error) {
        showError(`Lỗi kết nối: ${error.message}`);
    } finally {
        spinner.style.display = 'none';
        predictBtn.disabled = false;
    }
}

function displayResults(data) {
    const prediction = data.prediction;
    
    // Update prediction text
    document.getElementById('predictionClass').textContent = prediction.class_name;
    document.getElementById('predictionConfidence').textContent = 
        `Độ tin cậy: ${prediction.confidence_percent.toFixed(2)}% (${data.model_used})`;
    
    // Show only the top prediction as a single item
    const probList = document.getElementById('probabilityList');
    probList.innerHTML = '';
    probList.appendChild(createProbabilityItem(prediction.class_name, prediction.confidence, 1));
    
    // Hide chart-related content when only one prediction is shown
    const chartContainer = document.querySelector('.chart-container');
    if (chartContainer) {
        chartContainer.style.display = 'none';
    }
    
    // Show result section
    resultSection.style.display = 'block';
}

function createProbabilityItem(className, probability, rank) {
    const item = document.createElement('div');
    item.className = 'probability-item';
    
    const percent = (probability * 100).toFixed(1);
    
    item.innerHTML = `
        <span class="probability-rank">#${rank}</span>
        <span class="probability-label">${className}</span>
        <div class="probability-bar">
            <div class="probability-fill" style="width: ${percent}%">
                ${percent}%
            </div>
        </div>
        <span class="probability-percent">${percent}%</span>
    `;
    
    return item;
}

function drawChart(data) {
    const chartContainer = document.querySelector('.chart-container');
    if (chartContainer) {
        chartContainer.style.display = 'none';
    }

    if (chart) {
        chart.destroy();
        chart = null;
    }
}

// ==================== Reset ====================
function reset() {
    selectedImage = null;
    imageInput.value = '';
    previewSection.style.display = 'none';
    buttonGroup.style.display = 'none';
    resultSection.style.display = 'none';
    errorMessage.style.display = 'none';
    const chartContainer = document.querySelector('.chart-container');
    if (chartContainer) {
        chartContainer.style.display = 'none';
    }
    if (chart) {
        chart.destroy();
        chart = null;
    }
}

// ==================== Utilities ====================
function showError(message) {
    errorMessage.textContent = message;
    errorMessage.style.display = 'block';
}

async function checkStatus() {
    try {
        const response = await fetch('/api/status');
        const data = await response.json();
        
        if (!data.model_loaded) {
            showError('⚠️ Mô hình chưa được tải. Vui lòng chạy training trên Colab và download model!');
            predictBtn.disabled = true;
        }
        
        deviceInfo.textContent = `Device: ${data.device} | Classes: ${data.num_classes}`;
    } catch (error) {
        console.error('Error checking status:', error);
    }
}

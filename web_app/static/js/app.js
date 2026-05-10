// app.js - Frontend logic for clothing classifier (Multi-Model: ResNet + MobileNet)

let selectedImage = null;
let chart = null;

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

// Result DOM
const resnetPredictionClass = document.getElementById('resnetPredictionClass');
const resnetPredictionConfidence = document.getElementById('resnetPredictionConfidence');
const resnetError = document.getElementById('resnetError');

const mobilenetPredictionClass = document.getElementById('mobilenetPredictionClass');
const mobilenetPredictionConfidence = document.getElementById('mobilenetPredictionConfidence');
const mobilenetError = document.getElementById('mobilenetError');

// ==================== Initialize ====================
document.addEventListener('DOMContentLoaded', function () {
    setupEventListeners();
    checkStatus();
    clearResults();
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
}

// ==================== File Handling ====================
function handleFileSelect(event) {
    const file = event.target.files[0];
    if (file) processFile(file);
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
        if (file.type && file.type.startsWith('image/')) processFile(file);
        else showError('Vui lòng chọn tệp ảnh hợp lệ!');
    }
}

function handlePaste(event) {
    const clipboardItems = event.clipboardData?.items;
    if (!clipboardItems) return;

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
    if (file.size > 16 * 1024 * 1024) {
        showError('Kích thước tệp quá lớn (tối đa 16MB)!');
        return;
    }

    selectedImage = file;

    const reader = new FileReader();
    reader.onload = function (e) {
        previewImage.src = e.target.result;
        fileName.textContent = `📄 ${file.name}`;
        fileSize.textContent = `📊 ${formatFileSize(file.size)}`;

        previewSection.style.display = 'block';
        buttonGroup.style.display = 'flex';
        errorMessage.style.display = 'none';
        resultSection.style.display = 'none';
        clearResults();
    };
    reader.readAsDataURL(file);
}

function formatFileSize(bytes) {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return (Math.round((bytes / Math.pow(k, i)) * 100) / 100) + ' ' + sizes[i];
}

function clearResults() {
    resnetPredictionClass.textContent = '';
    resnetPredictionConfidence.textContent = '';
    mobilenetPredictionClass.textContent = '';
    mobilenetPredictionConfidence.textContent = '';

    resnetError.style.display = 'none';
    resnetError.textContent = '';
    mobilenetError.style.display = 'none';
    mobilenetError.textContent = '';
}

// ==================== Prediction ====================
async function predict() {
    if (!selectedImage) {
        showError('Vui lòng chọn ảnh!');
        return;
    }

    spinner.style.display = 'block';
    resultSection.style.display = 'none';
    errorMessage.style.display = 'none';
    predictBtn.disabled = true;

    try {
        const formData = new FormData();
        formData.append('image', selectedImage);

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

        if (data.success) displayResults(data);
        else showError(data.error || 'Lỗi dự đoán!');

    } catch (error) {
        showError(`Lỗi kết nối: ${error.message}`);
    } finally {
        spinner.style.display = 'none';
        predictBtn.disabled = false;
        resultSection.style.display = 'block';
    }
}

function displayResults(data) {
    const preds = data.predictions || {};
    const status = data.model_status || {};

        const timings = data.timings_ms || {};

    // ResNet
    if (preds.resnet) {
        const ms = timings.resnet_ms;
        resnetPredictionClass.textContent = preds.resnet.class_name;
        resnetPredictionConfidence.textContent = `Độ tin cậy: ${preds.resnet.confidence_percent.toFixed(2)}%`;
        if (typeof ms === 'number') resnetPredictionConfidence.textContent += ` | ${ms.toFixed(1)}ms`;
        resnetError.style.display = 'none';
        resnetError.textContent = '';
    } else {
        resnetPredictionClass.textContent = '—';
        resnetPredictionConfidence.textContent = '';
        const err = status?.resnet?.error || 'Model chưa sẵn sàng';
        resnetError.textContent = String(err);
        resnetError.style.display = 'block';
    }

    // MobileNet
    if (preds.mobilenet) {
        const ms = timings.mobilenet_ms;
        mobilenetPredictionClass.textContent = preds.mobilenet.class_name;
        mobilenetPredictionConfidence.textContent = `Độ tin cậy: ${preds.mobilenet.confidence_percent.toFixed(2)}%`;
        if (typeof ms === 'number') mobilenetPredictionConfidence.textContent += ` | ${ms.toFixed(1)}ms`;
        mobilenetError.style.display = 'none';
        mobilenetError.textContent = '';
    } else {
        mobilenetPredictionClass.textContent = '—';
        mobilenetPredictionConfidence.textContent = '';
        const err = status?.mobilenet?.error || 'Model chưa sẵn sàng';
        mobilenetError.textContent = String(err);
        mobilenetError.style.display = 'block';
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
    clearResults();
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

        if (!data || data.status !== 'ok') return;

        // If both models failed to load, disable predict.
        const resnetOk = !!data.model_status?.resnet?.loaded;
        const mobilenetOk = !!data.model_status?.mobilenet?.loaded;
        if (!resnetOk && !mobilenetOk) {
            showError('⚠️ Cả ResNet50 và MobileNet đều chưa tải được. Xem log/stack trace hiển thị trong ô lỗi.');
            predictBtn.disabled = true;
        }

        deviceInfo.textContent = `Device: ${data.device} | Classes: ${data.num_classes}`;

        const resnetText = resnetOk ? '✅ ResNet50 OK' : '❌ ResNet50 lỗi';
        const mobilenetText = mobilenetOk ? '✅ MobileNet OK' : '❌ MobileNet lỗi';
        modelInfo.textContent = `${resnetText} | ${mobilenetText}`;

    } catch (error) {
        console.error('Error checking status:', error);
    }
}


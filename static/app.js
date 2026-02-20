/**
 * AI Image Editor - Frontend JavaScript
 */

// API Base URL
const API_BASE = '';

// State
let currentMode = 'text-to-image';
let selectedModel = null;
let uploadedImages = [];
let models = [];

// DOM Elements
const modelSelect = document.getElementById('model-select');
const loadModelBtn = document.getElementById('load-model-btn');
const statusText = document.getElementById('status-text');
const modelInfo = document.getElementById('model-info');
const modelDescription = document.getElementById('model-description');
const modelType = document.getElementById('model-type');

const tabBtns = document.querySelectorAll('.tab-btn');
const textToImageSection = document.getElementById('text-to-image-section');
const imageToImageSection = document.getElementById('image-to-image-section');

const uploadArea = document.getElementById('upload-area');
const imageInput = document.getElementById('image-input');
const previewArea = document.getElementById('preview-area');
const previewContainer = document.getElementById('preview-container');
const clearImagesBtn = document.getElementById('clear-images-btn');

const t2iGenerateBtn = document.getElementById('t2i-generate-btn');
const i2iGenerateBtn = document.getElementById('i2i-generate-btn');

const resultSection = document.getElementById('result-section');
const resultImage = document.getElementById('result-image');
const downloadBtn = document.getElementById('download-btn');
const useAsInputBtn = document.getElementById('use-as-input-btn');

const loadingOverlay = document.getElementById('loading-overlay');
const loadingText = document.getElementById('loading-text');

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    loadModels();
    setupEventListeners();
    checkStatus();
});

// Load available models from API
async function loadModels() {
    try {
        const response = await fetch(`${API_BASE}/api/models`);
        const data = await response.json();
        models = data.models;
        
        // Populate model select
        modelSelect.innerHTML = '<option value="">-- Select a model --</option>';
        
        // Group models by type
        const textToImageModels = models.filter(m => m.type === 'text-to-image');
        const imageToImageModels = models.filter(m => m.type === 'image-to-image');
        
        if (textToImageModels.length > 0) {
            const optgroup = document.createElement('optgroup');
            optgroup.label = 'Text to Image';
            textToImageModels.forEach(model => {
                const option = document.createElement('option');
                option.value = model.id;
                option.textContent = model.name;
                optgroup.appendChild(option);
            });
            modelSelect.appendChild(optgroup);
        }
        
        if (imageToImageModels.length > 0) {
            const optgroup = document.createElement('optgroup');
            optgroup.label = 'Image to Image';
            imageToImageModels.forEach(model => {
                const option = document.createElement('option');
                option.value = model.id;
                option.textContent = model.name;
                optgroup.appendChild(option);
            });
            modelSelect.appendChild(optgroup);
        }
        
    } catch (error) {
        console.error('Failed to load models:', error);
        showStatus('Error loading models', 'error');
    }
}

// Check current model status
async function checkStatus() {
    try {
        const response = await fetch(`${API_BASE}/api/status`);
        const status = await response.json();
        updateStatusDisplay(status);
    } catch (error) {
        console.error('Failed to check status:', error);
    }
}

// Update status display
function updateStatusDisplay(status) {
    if (status.loaded && status.current_model) {
        statusText.textContent = `Model: ${status.current_model} (${status.model_type})`;
        statusText.parentElement.style.borderLeftColor = '#28a745';
        
        // Update model select
        modelSelect.value = status.current_model;
        selectedModel = models.find(m => m.id === status.current_model);
        
        // Show model info
        if (selectedModel) {
            modelInfo.classList.remove('hidden');
            modelDescription.textContent = selectedModel.description;
            modelType.textContent = `Type: ${selectedModel.type}`;
        }
    } else {
        statusText.textContent = 'No model loaded';
        statusText.parentElement.style.borderLeftColor = '#6c757d';
        modelInfo.classList.add('hidden');
    }
}

// Setup event listeners
function setupEventListeners() {
    // Model selection
    modelSelect.addEventListener('change', () => {
        const modelId = modelSelect.value;
        selectedModel = models.find(m => m.id === modelId);
        
        if (selectedModel) {
            modelInfo.classList.remove('hidden');
            modelDescription.textContent = selectedModel.description;
            modelType.textContent = `Type: ${selectedModel.type}`;
        } else {
            modelInfo.classList.add('hidden');
        }
    });
    
    loadModelBtn.addEventListener('click', loadSelectedModel);
    
    // Tab switching
    tabBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            const mode = btn.dataset.mode;
            switchMode(mode);
        });
    });
    
    // Image upload
    uploadArea.addEventListener('click', () => imageInput.click());
    uploadArea.addEventListener('dragover', handleDragOver);
    uploadArea.addEventListener('dragleave', handleDragLeave);
    uploadArea.addEventListener('drop', handleDrop);
    imageInput.addEventListener('change', handleFileSelect);
    clearImagesBtn.addEventListener('click', clearImages);
    
    // Generation buttons
    t2iGenerateBtn.addEventListener('click', generateTextToImage);
    i2iGenerateBtn.addEventListener('click', generateImageToImage);
    
    // Result actions
    downloadBtn.addEventListener('click', downloadResult);
    useAsInputBtn.addEventListener('click', useResultAsInput);
}

// Load selected model
async function loadSelectedModel() {
    const modelId = modelSelect.value;
    
    if (!modelId) {
        showStatus('Please select a model', 'error');
        return;
    }
    
    showLoading('Loading model...');
    
    try {
        const response = await fetch(`${API_BASE}/api/load_model`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ model_id: modelId })
        });
        
        const data = await response.json();
        
        if (response.ok) {
            showStatus(`Model ${modelId} loaded successfully`, 'success');
            updateStatusDisplay(data.status);
        } else {
            showStatus(`Error: ${data.error}`, 'error');
        }
    } catch (error) {
        console.error('Failed to load model:', error);
        showStatus('Failed to load model', 'error');
    } finally {
        hideLoading();
    }
}

// Switch between modes
function switchMode(mode) {
    currentMode = mode;
    
    // Update tabs
    tabBtns.forEach(btn => {
        btn.classList.toggle('active', btn.dataset.mode === mode);
    });
    
    // Show/hide sections
    textToImageSection.classList.toggle('hidden', mode !== 'text-to-image');
    imageToImageSection.classList.toggle('hidden', mode !== 'image-to-image');
}

// Drag and drop handlers
function handleDragOver(e) {
    e.preventDefault();
    uploadArea.classList.add('dragover');
}

function handleDragLeave(e) {
    e.preventDefault();
    uploadArea.classList.remove('dragover');
}

function handleDrop(e) {
    e.preventDefault();
    uploadArea.classList.remove('dragover');
    
    const files = e.dataTransfer.files;
    handleFiles(files);
}

function handleFileSelect(e) {
    const files = e.target.files;
    handleFiles(files);
}

// Handle uploaded files
function handleFiles(files) {
    for (const file of files) {
        if (file.type.startsWith('image/')) {
            const reader = new FileReader();
            reader.onload = (e) => {
                addImagePreview(e.target.result, file);
            };
            reader.readAsDataURL(file);
        }
    }
}

// Add image preview
function addImagePreview(dataUrl, file) {
    uploadedImages.push({
        dataUrl: dataUrl,
        file: file
    });
    
    updatePreviewArea();
}

// Update preview area
function updatePreviewArea() {
    previewContainer.innerHTML = '';
    
    uploadedImages.forEach((img, index) => {
        const item = document.createElement('div');
        item.className = 'preview-item';
        
        const imgEl = document.createElement('img');
        imgEl.src = img.dataUrl;
        
        const removeBtn = document.createElement('button');
        removeBtn.className = 'remove-btn';
        removeBtn.textContent = '×';
        removeBtn.onclick = () => removeImage(index);
        
        item.appendChild(imgEl);
        item.appendChild(removeBtn);
        previewContainer.appendChild(item);
    });
    
    previewArea.classList.toggle('hidden', uploadedImages.length === 0);
}

// Remove image
function removeImage(index) {
    uploadedImages.splice(index, 1);
    updatePreviewArea();
}

// Clear all images
function clearImages() {
    uploadedImages = [];
    updatePreviewArea();
    imageInput.value = '';
}

// Generate text to image
async function generateTextToImage() {
    const prompt = document.getElementById('t2i-prompt').value.trim();
    const width = parseInt(document.getElementById('t2i-width').value);
    const height = parseInt(document.getElementById('t2i-height').value);
    const steps = parseInt(document.getElementById('t2i-steps').value);
    
    if (!prompt) {
        showStatus('Please enter a prompt', 'error');
        return;
    }
    
    if (!selectedModel) {
        showStatus('Please select and load a model', 'error');
        return;
    }
    
    showLoading('Generating image...');
    
    try {
        const response = await fetch(`${API_BASE}/api/generate`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                model_id: selectedModel.id,
                prompt: prompt,
                width: width,
                height: height,
                steps: steps
            })
        });
        
        const data = await response.json();
        
        if (response.ok) {
            showResult(data.image);
            showStatus('Image generated successfully', 'success');
        } else {
            showStatus(`Error: ${data.error}`, 'error');
        }
    } catch (error) {
        console.error('Generation failed:', error);
        showStatus('Generation failed', 'error');
    } finally {
        hideLoading();
    }
}

// Generate image to image
async function generateImageToImage() {
    const prompt = document.getElementById('i2i-prompt').value.trim();
    const steps = parseInt(document.getElementById('i2i-steps').value);
    const guidance = parseFloat(document.getElementById('i2i-guidance').value);
    
    if (!prompt) {
        showStatus('Please enter a prompt', 'error');
        return;
    }
    
    if (uploadedImages.length === 0) {
        showStatus('Please upload at least one image', 'error');
        return;
    }
    
    if (!selectedModel) {
        showStatus('Please select and load a model', 'error');
        return;
    }
    
    showLoading('Processing image...');
    
    try {
        // Prepare images as base64
        const images = uploadedImages.map(img => img.dataUrl);
        
        const response = await fetch(`${API_BASE}/api/generate`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                model_id: selectedModel.id,
                prompt: prompt,
                images: images,
                steps: steps,
                guidance_scale: guidance
            })
        });
        
        const data = await response.json();
        
        if (response.ok) {
            showResult(data.image);
            showStatus('Image processed successfully', 'success');
        } else {
            showStatus(`Error: ${data.error}`, 'error');
        }
    } catch (error) {
        console.error('Processing failed:', error);
        showStatus('Processing failed', 'error');
    } finally {
        hideLoading();
    }
}

// Show result image
function showResult(imageData) {
    resultImage.src = imageData;
    resultSection.classList.remove('hidden');
    resultSection.scrollIntoView({ behavior: 'smooth' });
}

// Download result
function downloadResult() {
    const link = document.createElement('a');
    link.download = `generated_${Date.now()}.png`;
    link.href = resultImage.src;
    link.click();
}

// Use result as input
function useResultAsInput() {
    const dataUrl = resultImage.src;
    uploadedImages = [{
        dataUrl: dataUrl,
        file: null
    }];
    
    // Switch to image-to-image mode
    switchMode('image-to-image');
    updatePreviewArea();
}

// Show status message
function showStatus(message, type = 'info') {
    statusText.textContent = message;
    
    const colors = {
        success: '#28a745',
        error: '#dc3545',
        info: '#6c757d'
    };
    
    statusText.parentElement.style.borderLeftColor = colors[type] || colors.info;
}

// Loading overlay
function showLoading(message = 'Processing...') {
    loadingText.textContent = message;
    loadingOverlay.classList.remove('hidden');
}

function hideLoading() {
    loadingOverlay.classList.add('hidden');
}

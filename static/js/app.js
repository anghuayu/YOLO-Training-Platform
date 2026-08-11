/**
 * YOLO Training Platform - Core JavaScript
 * i18n system, API helpers, toast, modal, utilities
 */

// ============================================
// i18n - Internationalization System
// ============================================
let currentLang = 'zh';

const i18n = {
    zh: {
        // --- Nav ---
        dashboard: '仪表盘', datasets: '数据集', training: '模型训练',
        models: '模型管理', evaluate: '模型评估', mainMenu: '功能导航',
        platformSub: '智能训练平台 v1.0',

        // --- Dashboard ---
        dashboardTitle: '仪表盘', dashboardSub: '项目总览与快速操作',
        totalDatasets: '数据集', totalImages: '图片总数',
        totalAnnotated: '已标注', totalModels: '模型数量',
        recentTraining: '最近训练', viewAll: '查看全部',
        quickActions: '快速操作', newDataset: '新建数据集',
        startTraining: '开始训练', manageModels: '管理模型',
        evalModel: '评估模型', datasetOverview: '数据集概览',
        noTrainingRuns: '暂无训练记录', annotate: '标注',
        refresh: '刷新',

        // --- Datasets ---
        datasetsTitle: '数据集管理', datasetsSub: '创建和管理训练数据集',
        createDataset: '新建数据集', datasetName: '数据集名称',
        datasetNamePh: '例如：helmet_detection', description: '描述',
        descPh: '可选描述信息...', classesLabel: '类别标签（每行一个）',
        classesPh: 'helmet\nsafety_vest\nperson',
        classesHint: '每行输入一个类别名称，顺序决定类别 ID',
        cancel: '取消', create: '创建',
        noDatasets: '暂无数据集', noDatasetsDesc: '创建第一个数据集开始构建训练流程',
        annotated: '已标注', images: '图片', classes: '类别',
        view: '查看', upload: '上传', delete: '删除',
        uploadImages: '上传图片', uploadFiles: '选择图片文件',
        uploadFolder: '选择文件夹', uploading: '正在上传',
        uploadProgress: '上传进度',
        deleteDatasetConfirm: '确定删除数据集 "{name}"？此操作不可恢复。',
        exportYolo: '导出 YOLO 格式',
        noDesc: '暂无描述', noClasses: '未定义类别',

        // --- Annotation ---
        annotationTool: '标注工具', drawBoxes: '绘制目标框标注物体',
        smartAnnotate: '智能标注', save: '保存',
        searchImages: '搜索图片...', searchPlaceholder: '输入文件名搜索',
        searchImagesShort: '搜索...',
        drawBox: '画框', select: '选择', pan: '平移',
        zoomIn: '放大', zoomOut: '缩小', fitView: '适应窗口',
        prevImg: '上一张', nextImg: '下一张',
        classLabel: '类别:',
        labels: '标注列表', datasetClasses: '数据集类别',
        noAnnotations: '当前图片无标注',
        smartAnnotateTitle: '智能标注',
        smartDesc: '基于已标注图片训练临时模型，自动预测未标注图片',
        epochs: '训练轮数', confidence: '置信度阈值',
        startSmart: '开始智能标注',
        trainingModel: '正在训练临时模型...',
        predicting: '正在预测未标注图片...',
        smartCompleted: '智能标注完成',
        smartFailed: '智能标注失败',

        // --- Training ---
        trainingTitle: '模型训练', trainingSub: '配置并启动训练任务',
        trainConfig: '训练配置', modelType: '模型类型',
        selectDataset: '选择数据集...', selectModel: '选择预训练模型...',
        modelName: '模型名称', modelNamePh: '留空自动生成',
        pretrained: '预训练模型', trainFromScratch: '从头训练 / 默认',
        hyperparams: '超参数设置',
        epochsLabel: '训练轮数', imgSize: '图片尺寸',
        batchSize: '批量大小', learningRate: '学习率',
        patience: '早停耐心值', patienceHint: '连续 N 轮无提升则停止',
        optimizer: '优化器', device: '设备',
        deviceHint: '"0" 为第一张 GPU，"0,1" 多卡，"cpu" 用处理器',
        startTrain: '开始训练',
        activeTraining: '正在训练', stop: '停止',
        trainingHistory: '训练历史',
        trainingCurves: '训练曲线',
        noTrainingYet: '暂无训练记录',
        selectDatasetWarn: '请先选择数据集',
        trainingStarted: '训练已启动',
        stopConfirm: '确定停止当前训练？',
        trainingStopped: '训练已停止',
        trainingCompleted: '训练完成',
        trainingFailed: '训练失败',
        deleteRunConfirm: '确定删除此训练记录？',

        // --- Models ---
        modelsTitle: '模型管理', modelsSub: '管理、导出和下载训练好的模型',
        noModels: '暂无模型', noModelsDesc: '训练第一个模型后会在这里显示',
        exportModel: '导出模型', exportFormat: '导出格式',
        exportImgSize: '图片尺寸', halfPrecision: '半精度 (FP16)',
        simplifyModel: '简化模型', exporting: '正在导出...',
        exportSuccess: '模型导出成功', exportFailed: '导出失败',
        editModel: '编辑模型', editName: '模型名称',
        editDesc: '描述', modelUpdated: '模型已更新',
        deleteModelConfirm: '确定删除模型 "{name}"？此操作不可恢复。',
        modelDeleted: '模型已删除',
        exportedFiles: '已导出文件', download: '下载',

        // --- Evaluate ---
        evalTitle: '模型评估', evalSub: '在数据集上评估模型性能',
        evalSettings: '评估设置', selectModelWarn: '请选择模型',
        selectBothWarn: '请同时选择模型和数据集',
        runEval: '开始评估', evaluating: '正在评估模型，请稍候...',
        evalComplete: '评估完成', evalFailed: '评估失败',
        perClassMap: '各类别 mAP@50', metricsOverview: '指标概览',
        perClassResults: '各类别详细结果', classId: '类别 ID',
        className: '类别名称', precision: '精确率', recall: '召回率',
        evalHistory: '评估历史',
        noEvalYet: '运行评估后结果会显示在这里',

        // --- Common ---
        confirm: '确定', close: '关闭',
        loading: '加载中...', error: '错误',
        success: '成功', warning: '警告', info: '提示',
        saved: '已保存', deleted: '已删除',
        failed: '失败', completed: '已完成',
    },
    en: {
        dashboard: 'Dashboard', datasets: 'Datasets', training: 'Training',
        models: 'Models', evaluate: 'Evaluate', mainMenu: 'Navigation',
        platformSub: 'Training Platform v1.0',

        dashboardTitle: 'Dashboard', dashboardSub: 'Project overview & quick actions',
        totalDatasets: 'Datasets', totalImages: 'Total Images',
        totalAnnotated: 'Annotated', totalModels: 'Models',
        recentTraining: 'Recent Training', viewAll: 'View All',
        quickActions: 'Quick Actions', newDataset: 'New Dataset',
        startTraining: 'Start Training', manageModels: 'Manage Models',
        evalModel: 'Evaluate Model', datasetOverview: 'Dataset Overview',
        noTrainingRuns: 'No training runs yet', annotate: 'Annotate',
        refresh: 'Refresh',

        datasetsTitle: 'Datasets', datasetsSub: 'Create and manage training datasets',
        createDataset: 'New Dataset', datasetName: 'Dataset Name',
        datasetNamePh: 'e.g., helmet_detection', description: 'Description',
        descPh: 'Optional description...', classesLabel: 'Classes (one per line)',
        classesPh: 'helmet\nsafety_vest\nperson',
        classesHint: 'Enter each class name on a new line. Order determines class ID.',
        cancel: 'Cancel', create: 'Create',
        noDatasets: 'No Datasets Yet', noDatasetsDesc: 'Create your first dataset to start building your training pipeline.',
        annotated: 'Annotated', images: 'Images', classes: 'Classes',
        view: 'View', upload: 'Upload', delete: 'Delete',
        uploadImages: 'Upload Images', uploadFiles: 'Select Files',
        uploadFolder: 'Select Folder', uploading: 'Uploading...',
        uploadProgress: 'Upload Progress',
        deleteDatasetConfirm: 'Delete dataset "{name}"? This cannot be undone.',
        exportYolo: 'Export YOLO Format',
        noDesc: 'No description', noClasses: 'No classes defined',

        annotationTool: 'Annotation Tool', drawBoxes: 'Draw bounding boxes to annotate objects',
        smartAnnotate: 'Smart Annotate', save: 'Save',
        searchImages: 'Search images...', searchPlaceholder: 'Type to filter by filename',
        searchImagesShort: 'Search...',
        drawBox: 'Draw', select: 'Select', pan: 'Pan',
        zoomIn: 'Zoom In', zoomOut: 'Zoom Out', fitView: 'Fit',
        prevImg: 'Previous', nextImg: 'Next',
        classLabel: 'Class:',
        labels: 'Labels', datasetClasses: 'Dataset Classes',
        noAnnotations: 'No annotations on this image',
        smartAnnotateTitle: 'Smart Annotation',
        smartDesc: 'Train a temporary model on labeled data, then auto-predict remaining images',
        epochs: 'Epochs', confidence: 'Confidence',
        startSmart: 'Start Smart Annotation',
        trainingModel: 'Training temporary model...',
        predicting: 'Predicting unannotated images...',
        smartCompleted: 'Smart annotation completed',
        smartFailed: 'Smart annotation failed',

        trainingTitle: 'Model Training', trainingSub: 'Configure and launch training runs',
        trainConfig: 'Training Configuration', modelType: 'Model Type',
        selectDataset: 'Select a dataset...', selectModel: 'Select pretrained model...',
        modelName: 'Model Name', modelNamePh: 'Auto-generated if empty',
        pretrained: 'Pretrained Model', trainFromScratch: 'Train from scratch / default',
        hyperparams: 'Hyperparameters',
        epochsLabel: 'Epochs', imgSize: 'Image Size',
        batchSize: 'Batch Size', learningRate: 'Learning Rate',
        patience: 'Patience', patienceHint: 'Early stop after N epochs without improvement',
        optimizer: 'Optimizer', device: 'Device',
        deviceHint: '"0" for first GPU, "0,1" multi-GPU, "cpu" for CPU',
        startTrain: 'Start Training',
        activeTraining: 'Active Training', stop: 'Stop',
        trainingHistory: 'Training History',
        trainingCurves: 'Training Curves',
        noTrainingYet: 'No training runs yet',
        selectDatasetWarn: 'Please select a dataset',
        trainingStarted: 'Training started',
        stopConfirm: 'Stop this training run?',
        trainingStopped: 'Training stopped',
        trainingCompleted: 'Training completed',
        trainingFailed: 'Training failed',
        deleteRunConfirm: 'Delete this training run?',

        modelsTitle: 'Model Management', modelsSub: 'Manage, export and download trained models',
        noModels: 'No Models Yet', noModelsDesc: 'Train your first model to see it here.',
        exportModel: 'Export Model', exportFormat: 'Export Format',
        exportImgSize: 'Image Size', halfPrecision: 'Half Precision (FP16)',
        simplifyModel: 'Simplify Model', exporting: 'Exporting...',
        exportSuccess: 'Model exported successfully', exportFailed: 'Export failed',
        editModel: 'Edit Model', editName: 'Model Name',
        editDesc: 'Description', modelUpdated: 'Model updated',
        deleteModelConfirm: 'Delete model "{name}"? This cannot be undone.',
        modelDeleted: 'Model deleted',
        exportedFiles: 'Exported Files', download: 'Download',

        evalTitle: 'Model Evaluation', evalSub: 'Evaluate model performance on datasets',
        evalSettings: 'Evaluation Settings', selectModelWarn: 'Please select a model',
        selectBothWarn: 'Please select both a model and a dataset',
        runEval: 'Run Evaluation', evaluating: 'Evaluating model, please wait...',
        evalComplete: 'Evaluation complete', evalFailed: 'Evaluation failed',
        perClassMap: 'Per-Class mAP@50', metricsOverview: 'Metrics Overview',
        perClassResults: 'Per-Class Results', classId: 'Class ID',
        className: 'Class Name', precision: 'Precision', recall: 'Recall',
        evalHistory: 'Evaluation History',
        noEvalYet: 'Run an evaluation to see results here',

        confirm: 'Confirm', close: 'Close',
        loading: 'Loading...', error: 'Error',
        success: 'Success', warning: 'Warning', info: 'Info',
        saved: 'Saved', deleted: 'Deleted',
        failed: 'Failed', completed: 'Completed',
    }
};

function t(key) {
    return (i18n[currentLang] && i18n[currentLang][key]) || i18n['en'][key] || key;
}

function toggleLang() {
    currentLang = currentLang === 'zh' ? 'en' : 'zh';
    applyTranslations();
    const btn = document.getElementById('lang-toggle');
    if (btn) btn.textContent = currentLang === 'zh' ? 'EN' : '中';
}

function applyTranslations() {
    document.querySelectorAll('[data-i18n]').forEach(el => {
        const key = el.getAttribute('data-i18n');
        const text = t(key);
        if (text) el.textContent = text;
    });
    document.querySelectorAll('[data-i18n-ph]').forEach(el => {
        const key = el.getAttribute('data-i18n-ph');
        const text = t(key);
        if (text) el.placeholder = text;
    });
    document.querySelectorAll('[data-i18n-title]').forEach(el => {
        const key = el.getAttribute('data-i18n-title');
        const text = t(key);
        if (text) el.title = text;
    });
}

// ============================================
// Helper to extract error message from FastAPI error response
function _extractError(err) {
    if (!err) return 'Unknown error';
    const detail = err.detail;
    if (!detail) return err.message || 'Unknown error';
    if (typeof detail === 'string') return detail;
    if (Array.isArray(detail)) return detail.map(e => e.msg || e.message || JSON.stringify(e)).join('; ');
    if (typeof detail === 'object') return detail.msg || detail.message || JSON.stringify(detail);
    return String(detail);
}

// API Helper (with upload progress support)
// ============================================
const API = {
    async get(url) {
        const res = await fetch(url);
        if (!res.ok) {
            const err = await res.json().catch(() => ({ detail: res.statusText }));
            throw new Error(_extractError(err));
        }
        return res.json();
    },
    async post(url, data) {
        const res = await fetch(url, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(data),
        });
        if (!res.ok) {
            const err = await res.json().catch(() => ({ detail: res.statusText }));
            throw new Error(_extractError(err));
        }
        return res.json();
    },
    async put(url, data) {
        const res = await fetch(url, {
            method: 'PUT',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(data),
        });
        if (!res.ok) {
            const err = await res.json().catch(() => ({ detail: res.statusText }));
            throw new Error(_extractError(err));
        }
        return res.json();
    },
    async delete(url) {
        const res = await fetch(url, { method: 'DELETE' });
        if (!res.ok) {
            const err = await res.json().catch(() => ({ detail: res.statusText }));
            throw new Error(_extractError(err));
        }
        return res.json();
    },
    async upload(url, formData, onProgress) {
        return new Promise((resolve, reject) => {
            const xhr = new XMLHttpRequest();
            xhr.open('POST', url);
            xhr.onload = () => {
                if (xhr.status >= 200 && xhr.status < 300) {
                    try { resolve(JSON.parse(xhr.responseText)); }
                    catch { resolve(xhr.responseText); }
                } else {
                    try {
                        const err = JSON.parse(xhr.responseText);
                        reject(new Error(err.detail || `HTTP ${xhr.status}`));
                    } catch { reject(new Error(`HTTP ${xhr.status}`)); }
                }
            };
            xhr.onerror = () => reject(new Error('Network error'));
            if (xhr.upload && onProgress) {
                xhr.upload.onprogress = (e) => {
                    if (e.lengthComputable) onProgress(Math.round(e.loaded / e.total * 100));
                };
            }
            xhr.send(formData);
        });
    },
};

// ============================================
// Toast Notifications
// ============================================
function showToast(message, type = 'info') {
    let container = document.querySelector('.toast-container');
    if (!container) {
        container = document.createElement('div');
        container.className = 'toast-container';
        document.body.appendChild(container);
    }
    const toast = document.createElement('div');
    toast.className = `toast ${type}`;
    const icons = {
        success: 'fa-check-circle', error: 'fa-exclamation-circle',
        warning: 'fa-exclamation-triangle', info: 'fa-info-circle',
    };
    const colors = { success: 'var(--success)', error: 'var(--danger)', warning: 'var(--warning)', info: 'var(--primary)' };
    toast.innerHTML = `
        <i class="fas ${icons[type] || icons.info}" style="color: ${colors[type] || colors.info}"></i>
        <span class="toast-msg">${message}</span>
    `;
    container.appendChild(toast);
    setTimeout(() => {
        toast.style.opacity = '0';
        toast.style.transform = 'translateX(100%)';
        toast.style.transition = 'all 0.3s ease';
        setTimeout(() => toast.remove(), 300);
    }, 3500);
}

// ============================================
// Modal
// ============================================
function openModal(id) {
    const modal = document.getElementById(id);
    if (modal) modal.classList.add('active');
}
function closeModal(id) {
    const modal = document.getElementById(id);
    if (modal) modal.classList.remove('active');
}
document.addEventListener('click', (e) => {
    if (e.target.classList.contains('modal-overlay')) e.target.classList.remove('active');
});
document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape') document.querySelectorAll('.modal-overlay.active').forEach(m => m.classList.remove('active'));
});

// ============================================
// Progress Overlay (full-screen loading with progress bar)
// ============================================
function showProgressOverlay(title, subtitle) {
    let overlay = document.getElementById('progress-overlay');
    if (!overlay) {
        overlay = document.createElement('div');
        overlay.id = 'progress-overlay';
        overlay.className = 'progress-overlay';
        document.body.appendChild(overlay);
    }
    overlay.innerHTML = `
        <div class="progress-overlay-content">
            <div class="progress-spinner"></div>
            <h3 id="progress-title">${title}</h3>
            <p id="progress-subtitle">${subtitle || ''}</p>
            <div class="progress-bar" style="width: 280px; height: 10px; margin-top: 16px;">
                <div class="progress-fill" id="progress-bar-fill" style="width: 0%;"></div>
            </div>
            <div id="progress-percent" style="margin-top: 8px; font-size: 14px; color: var(--text-secondary);">0%</div>
        </div>
    `;
    overlay.classList.add('active');
}

function updateProgressOverlay(percent, subtitle) {
    const fill = document.getElementById('progress-bar-fill');
    const pct = document.getElementById('progress-percent');
    const sub = document.getElementById('progress-subtitle');
    if (fill) fill.style.width = percent + '%';
    if (pct) pct.textContent = Math.round(percent) + '%';
    if (sub && subtitle) sub.textContent = subtitle;
}

function hideProgressOverlay() {
    const overlay = document.getElementById('progress-overlay');
    if (overlay) overlay.classList.remove('active');
}

// ============================================
// Utility Functions
// ============================================
function formatSize(bytes) {
    if (bytes < 1024) return bytes + ' B';
    if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + ' KB';
    return (bytes / (1024 * 1024)).toFixed(1) + ' MB';
}

function formatDate(dateStr) {
    if (!dateStr) return '-';
    try {
        const d = new Date(dateStr);
        const locale = currentLang === 'zh' ? 'zh-CN' : 'en-US';
        return d.toLocaleDateString(locale) + ' ' + d.toLocaleTimeString(locale, { hour: '2-digit', minute: '2-digit' });
    } catch { return dateStr; }
}

function formatNumber(num) {
    if (num >= 1000000) return (num / 1000000).toFixed(1) + 'M';
    if (num >= 1000) return (num / 1000).toFixed(1) + 'K';
    return num.toString();
}

function statusBadge(status) {
    const zhMap = {
        'training': { text: '训练中', icon: 'fa-spinner fa-spin' },
        'completed': { text: '已完成', icon: 'fa-check' },
        'stopped': { text: '已停止', icon: 'fa-stop' },
        'error': { text: '错误', icon: 'fa-exclamation' },
        'queued': { text: '排队中', icon: 'fa-clock' },
    };
    const enMap = {
        'training': { text: 'Training', icon: 'fa-spinner fa-spin' },
        'completed': { text: 'Completed', icon: 'fa-check' },
        'stopped': { text: 'Stopped', icon: 'fa-stop' },
        'error': { text: 'Error', icon: 'fa-exclamation' },
        'queued': { text: 'Queued', icon: 'fa-clock' },
    };
    const clsMap = {
        'training': 'badge-primary', 'completed': 'badge-success',
        'stopped': 'badge-warning', 'error': 'badge-danger', 'queued': 'badge-info',
    };
    const map = currentLang === 'zh' ? zhMap : enMap;
    const s = map[status] || { text: status, icon: 'fa-question' };
    const cls = clsMap[status] || 'badge-info';
    return `<span class="badge ${cls}"><i class="fas ${s.icon}"></i> ${s.text}</span>`;
}

function confirmAction(message) {
    return new Promise((resolve) => {
        if (confirm(message)) resolve(true);
        else resolve(false);
    });
}

// ============================================
// Sidebar HTML
// ============================================
function getSidebarHTML(currentPage) {
    const items = [
        { href: '/', icon: 'fa-th-large', labelKey: 'dashboard', page: 'dashboard' },
        { href: '/datasets', icon: 'fa-database', labelKey: 'datasets', page: 'datasets' },
        { href: '/train', icon: 'fa-brain', labelKey: 'training', page: 'train' },
        { href: '/models', icon: 'fa-cube', labelKey: 'models', page: 'models' },
        { href: '/evaluate', icon: 'fa-chart-bar', labelKey: 'evaluate', page: 'evaluate' },
    ];
    const langLabel = currentLang === 'zh' ? 'EN' : '中';

    return `
    <aside class="sidebar">
        <div class="sidebar-brand">
            <div class="brand-icon"><i class="fas fa-eye"></i></div>
            <div class="brand-info">
                <div class="brand-text">YOLO Platform</div>
                <div class="brand-sub" data-i18n="platformSub">${t('platformSub')}</div>
            </div>
            <button id="lang-toggle" class="lang-btn" onclick="toggleLang()" title="Switch Language">${langLabel}</button>
        </div>
        <nav class="sidebar-nav">
            <div class="nav-section">
                <div class="nav-section-title" data-i18n="mainMenu">${t('mainMenu')}</div>
                ${items.map(item => `
                    <a href="${item.href}" class="nav-item ${currentPage === item.page ? 'active' : ''}" data-page="${item.page}">
                        <i class="fas ${item.icon}"></i>
                        <span data-i18n="${item.labelKey}">${t(item.labelKey)}</span>
                    </a>
                `).join('')}
            </div>
        </nav>
        <div class="sidebar-footer">
            <div class="sidebar-footer-text">YOLO Training Platform</div>
        </div>
    </aside>`;
}

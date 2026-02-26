/**
 * Minimal microphone recorder for the Voice Cloning Web UI.
 * Uses getUserMedia + MediaRecorder to capture audio, then sends
 * it to the server via HTMX-compatible form submission.
 */

let mediaRecorder = null;
let audioChunks = [];

async function toggleRecording(btn) {
    if (mediaRecorder && mediaRecorder.state === 'recording') {
        stopRecording(btn);
    } else {
        await startRecording(btn);
    }
}

async function startRecording(btn) {
    try {
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        mediaRecorder = new MediaRecorder(stream, { mimeType: 'audio/webm' });
        audioChunks = [];

        mediaRecorder.ondataavailable = (e) => {
            if (e.data.size > 0) audioChunks.push(e.data);
        };

        mediaRecorder.onstop = () => {
            stream.getTracks().forEach(t => t.stop());
            const blob = new Blob(audioChunks, { type: 'audio/webm' });
            sendRecording(blob);
        };

        mediaRecorder.start();

        // UI feedback
        btn.classList.add('btn-error');
        btn.classList.remove('btn-secondary');
        document.getElementById('mic-btn-text').textContent = 'Stop Recording';
        document.getElementById('recording-indicator').classList.remove('hidden');
    } catch (err) {
        console.error('Microphone access denied:', err);
        alert('Microphone access denied. Please allow microphone access and try again.');
    }
}

function stopRecording(btn) {
    if (mediaRecorder && mediaRecorder.state === 'recording') {
        mediaRecorder.stop();
    }

    // UI feedback
    btn.classList.remove('btn-error');
    btn.classList.add('btn-secondary');
    document.getElementById('mic-btn-text').textContent = 'Record from Microphone';
    document.getElementById('recording-indicator').classList.add('hidden');
}

function sendRecording(blob) {
    const formData = new FormData();
    formData.append('audio', blob, 'recording.webm');

    htmx.ajax('POST', '/api/upload-recording', {
        target: '#embedding-result',
        swap: 'innerHTML',
        values: formData,
    });
}

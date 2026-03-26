const video = document.getElementById('videoFeed');
const textOutput = document.getElementById('textOutput');
const voiceBtn = document.getElementById('voiceBtn');

// Access WebCam
async function setupCamera() {
    const stream = await navigator.mediaDevices.getUserMedia({ video: true });
    video.srcObject = stream;
}

// Function to send frame to Python Backend
async function sendFrame() {
    const canvas = document.createElement('canvas');
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    canvas.getContext('2d').drawImage(video, 0, 0);
    
    // Convert to Base64 to send via JSON
    const dataUrl = canvas.toDataURL('image/jpeg');

    try {
        const response = await fetch('/predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ image: dataUrl })
        });
        const data = await response.json();
        textOutput.innerText = data.prediction;
    } catch (err) {
        console.error("Error connecting to backend:", err);
    }
}

// Voice Translation (Text-to-Speech)
voiceBtn.addEventListener('click', () => {
    const msg = new SpeechSynthesisUtterance(textOutput.innerText);
    window.speechSynthesis.speak(msg);
});

setupCamera();
// Run prediction every 500ms (adjust based on your RNN sequence length)
setInterval(sendFrame, 500);
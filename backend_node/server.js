const express = require('express');
const cors = require('cors');
const WebSocket = require('ws');
const http = require('http');

const app = express();
app.use(cors());
app.use(express.json());

const server = http.createServer(app);
const wss = new WebSocket.Server({ server });

// Basic API route
app.get('/api/status', (req, res) => {
    res.json({ status: "Node.js Backend is running", ai_bridge: "active" });
});

// WebSocket Server to bridge React with Python
wss.on('connection', (ws) => {
    console.log('React Client connected to WebSockets');

    ws.on('message', (message) => {
        // Here we will eventually proxy the webcam frames (message)
        // over to the Python ML server (e.g. over ZeroMQ or an HTTP call)
        // and return the prediction text.
        
        // For now, this is a placeholder echo.
        const msgStr = message.toString();
        if (msgStr === "TEST") {
             ws.send(JSON.stringify({ type: "PREDICTION", text: "ISL Recognition Online" }));
        }
    });

    ws.on('close', () => {
        console.log('React Client disconnected');
    });
});

const PORT = 5000;
server.listen(PORT, () => {
    console.log(`Node.js API and WebSocket Server running on port ${PORT}`);
});

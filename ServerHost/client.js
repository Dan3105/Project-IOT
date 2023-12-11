const path = require('path');
const express = require('express');
const WebSocket = require('ws');

const app = express();

//const WS_PORT = 8888; //Port for server
const HTTP_PORT = 8000; //Port for client

// const wsServer = new WebSocket.Server({port: WS_PORT}, () => console.log(`WS server is listening on port: ${WS_PORT}`));

// let connectedClients = []
// wsServer.on('connection', (ws, req) => {
//     console.log(`Connected ${ws}`);
//     connectedClients.push(ws);

//     ws.on('message', data => {
//         //handler data processing in here

//         connectedClients.forEach((ws, i) => {
//             if(ws.readyState === ws.OPEN){

//                 ws.send(data);
//             }else{
//                 connectedClients.splice(i ,1);
//             }
//         })
//     })
// });

app.get('/client', (req, res) => res.sendFile(path.resolve(__dirname, './client.html')));
app.listen(HTTP_PORT, () => console.log(`HTTP server is listening at ${HTTP_PORT}`));
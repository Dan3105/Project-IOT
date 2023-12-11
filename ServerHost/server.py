import asyncio
import websockets
from model_detection import ModelDetection, decode_image, encode_image

IP = "192.168.1.35" #ipv4
PORT = 8888

connected_clients = dict()
model_detection = ModelDetection()
print('Model has been loaded')

async def handle_client(websocket, path):
    print(f'New Connection address {websocket.remote_address[0]}')

    #each time a client connect to this server, the connect_client will save the {client_address: client_socket}
    connected_clients[websocket.remote_address[0]] = websocket
    try:
        while True:
            # Whenever the client sent data to this server (we know that is only arduino send data to this server)
            # Receive raw video data from the client
            data = await websocket.recv() #file binary format
    
            detect_data, _ishavingperson = model_detection.predict(data)
            if _ishavingperson:
                print('Data detect human')

            # just send the data to all currently connected client from this server
            for (addr, client_socket) in connected_clients.items():
                if not client_socket.closed:
                    asyncio.gather(client_socket.send(detect_data))    
                else:
                    print(f'Lose Connection of this {addr}')
                    del connected_clients[addr]        
    except websockets.exceptions.ConnectionClosed as e:
        print(e)
        pass
    finally:
        print(f'Lose Connection of this {websocket.remote_address[0]}')
        del connected_clients[websocket.remote_address[0]]

server = websockets.serve(handle_client, IP, PORT)


def start_server():
    print(f"Run ws in host: {IP} at port {PORT}")
    asyncio.get_event_loop().run_until_complete(server)
    asyncio.get_event_loop().run_forever()

if __name__ == "__main__":
    start_server()

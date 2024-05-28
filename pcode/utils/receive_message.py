import pickle
from socket import *
import os
import time


def receive_message(LOCAl_IP, PORT, BUFLEN, file_path=r"./recevied_file"):
    holding_flag = True  # check whether to connect continuously
    message_lst = []
    while holding_flag:
        listenSocket = socket(AF_INET, SOCK_STREAM)
        sleep_sec = 1
        while sleep_sec < 10:
            try:
                listenSocket.bind((LOCAl_IP, PORT))
                break
            except:
                print("address may be already in use")
                time.sleep(sleep_sec)
                sleep_sec += 1

        listenSocket.listen(2)
        print(f'waiting for connetion at port {PORT}...')

        dataSocket, addr = listenSocket.accept()
        print('accept a connection:', addr)

        # flagData is used to check whether the message is a file or a text
        flagData = dataSocket.recv(BUFLEN).decode()
        print("flagData", flagData)
        dataSocket.send("flag data received success".encode())

        start_time = time.time()
        SEPARATOR = "<SEPARATOR>"
        if "is_file" in flagData:
            while True:
                headDate = dataSocket.recv(BUFLEN).decode()
                if headDate == "exit":
                    break

                if not os.path.exists(file_path):
                    os.makedirs(file_path)
                file_name, message_size = headDate.split(SEPARATOR)
                file_name = os.path.basename(file_name)
                file_path_name = os.path.join(file_path, file_name)
                message_lst.append(file_path_name)
                message_size = int(message_size)
                dataSocket.send("message head received success".encode())

                received_length, epoch = 0, 0
                print(f"{file_name}, size:{str(message_size)}")
                file_write = open(file_path_name, "wb")
                print("Start receiving the message")
                while True:
                    file_date = dataSocket.recv(BUFLEN)
                    file_write.write(file_date)
                    received_length += len(file_date)
                    if epoch % 10000 == 0:
                        print(message_size, "---", received_length, "---", len(file_date))
                        print(f"have received：{received_length}/{message_size}，epoch：{epoch}")
                    if received_length >= message_size:
                        break
                    epoch += 1
                print("Finish receiving the message")
                end_time = time.time()
                print('cost %f second' % (end_time - start_time))
                dataSocket.send("message data received success".encode())
            holding_flag = False

        else:
            while True:
                if "pickle" in flagData:
                    messageData = dataSocket.recv(BUFLEN)
                    messageData = pickle.loads(messageData)
                else:
                    messageData = dataSocket.recv(BUFLEN).decode()
                print("messageData", messageData, type(messageData))
                end_time = time.time()
                print('cost %f second' % (end_time - start_time))
                dataSocket.send("message data received success".encode())
                if isinstance(messageData, str) and messageData == "exit":
                    break
                elif isinstance(messageData, str) and messageData == "close":
                    holding_flag = False
                message_lst.append(messageData)
        # dataSocket.shutdown(2)
        dataSocket.close()
        listenSocket.shutdown(2)
        listenSocket.close()

    return message_lst

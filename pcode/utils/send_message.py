# coding=utf-8
import pickle
from socket import *
import os
import time


def send_message(TARGET_IP, TARGET_PORT, BUFLEN, is_file=False, file_name="", message=None, is_pickle=False):
    print("sending message:", message)
    dataSocket = socket(AF_INET, SOCK_STREAM)
    while True:
        try:
            dataSocket.connect((TARGET_IP, TARGET_PORT))
            break
        except:
            time.sleep(3)
            print("can not connect, try again")

    # send file
    if is_file:
        try:
            message_size = os.path.getsize(file_name)
        except Exception as e:
            print("no such file")
            print("exception:", e)

        dataSocket.send(f"is_file".encode())
        flagSuccess = dataSocket.recv(BUFLEN).decode()
        print("receieve flag:", flagSuccess)

        SEPARATOR = "<SEPARATOR>"
        dataSocket.send(f"{file_name}{SEPARATOR}{message_size}".encode())
        headSuccess = dataSocket.recv(BUFLEN).decode()
        print("send head:", headSuccess)

        message = open(file_name, "rb")
        dataSocket.sendall(message.read())
        dateSuccess = dataSocket.recv(BUFLEN).decode()
        print("send data:", dateSuccess)

    # send text
    else:
        if is_pickle:
            dataSocket.send(f"is_not_file_is_pickle".encode())
        else:
            dataSocket.send(f"is_not_file".encode())
        flagSuccess = dataSocket.recv(BUFLEN).decode()
        print("send flag:", flagSuccess)

        if is_pickle:
            dataSocket.send(message)
        else:
            dataSocket.send(message.encode())
        dateSuccess = dataSocket.recv(BUFLEN).decode()
        print("send data:", dateSuccess)

    if is_pickle:
        dataSocket.send(pickle.dumps("exit"))
    else:
        dataSocket.send("exit".encode())
    dataSocket.shutdown(2)
    dataSocket.close()


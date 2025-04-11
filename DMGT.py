import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
import threading
import time

class ElevatorModel(nn.Module):
    def _init_(self):
        super(ElevatorModel, self)._init_()
        self.fc1 = nn.Linear(4, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 3)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class PriorityQueue:
    def _init_(self):
        self.queue = []

    def put(self, item, priority):
        self.queue.append((priority, item))
        self.queue.sort(reverse=True)

    def get(self):
        return self.queue.pop(0)[1] if self.queue else None

class ElevatorThread(threading.Thread):
    def _init_(self, elevator_id, request_queue, model):
        threading.Thread._init_(self)
        self.elevator_id = elevator_id
        self.request_queue = request_queue
        self.model = model

    def run(self):
        while True:
            request = self.request_queue.get()
            if request is None:  
                break
            print(f"Elevator {self.elevator_id} processing request: {request}")
            time.sleep(1)
cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
button_active = True
graph = {1: {2: 2, 3: 4}, 2: {3: 1, 4: 7}, 3: {4: 3}, 4: {}}
data = np.array([[1, 5, 3, 10], [4, 9, 2, 5], [2, 7, 1, 15]])
labels = np.array([0, 1, 2])
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
model = ElevatorModel()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
epochs = 100
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)
    loss.backward()
    optimizer.step()
    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')
request_queue = PriorityQueue()
num_elevators = 2 
elevator_threads = []
for i in range(num_elevators):
    thread = ElevatorThread(i, request_queue, model)
    thread.start()
    elevator_threads.append(thread)
while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    if len(faces) == 0:
        button_active = False
        print("Lift call button is deactivated.")
    else:
        button_active = True
        print("Lift call button is active.")
    cv2.imshow('Original Frame', frame)
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
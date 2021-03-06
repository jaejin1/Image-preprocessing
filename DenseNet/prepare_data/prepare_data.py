import cv2
import os
import numpy as np
import pickle

cascPath = 'haarcascade_frontalface_default.xml'
faceCascade = cv2.CascadeClassifier(cascPath)

count = 1

# real image

for fn in sorted(os.listdir('celeba_data')):
    image = cv2.imread('celeba_data/' + fn)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(gray, 5, 5)

    if len(faces) == 0:
        pass
    else:

        x, y, w, h = faces[0]
        image_crop = image[y:y+w, x:x+w, :]
        image_resize = cv2.resize(image_crop, (32, 32))

        if count == 1:
            data = np.array([image_resize])
        else:
            data = np.append(data,[image_resize], axis=0)

        print(data.shape)

        if count == 55000:
            break
        else:
            count += 1

# fake image

for fn in sorted(os.listdir('began_data')):
    image = cv2.imread('began_data/' + fn)
    image_resize = cv2.resize(image, (32, 32))

    data = np.append(data, [image_resize], axis=0)

    print(data.shape)

    if count == 109999:
        break
    else:
        count += 1

with open('data.pkl', 'wb') as f:
    pickle.dump(data, f)
print('data.pkl save!')

# label data

for i in range(110000):
    if i == 0:
        label_data = np.array([[1,0]])
    elif i < 55000:
        label_data = np.append(label_data, [[1, 0]], axis=0)
    else:
        label_data = np.append(label_data, [[0, 1]], axis=0)

with open('label_data.pkl', 'wb') as f:
    pickle.dump(label_data, f)
print('label_data.pkl save!')

# Shuffling data

indices = np.random.permutation(len(data))

data = data[indices]
label_data = label_data[indices]

# save data

with open('label_data.pkl', 'wb') as f:
    pickle.dump(label_data, f)

with open('data.pkl', 'wb') as f:
    pickle.dump(data, f)

print('finish!!')

# Recognition Engine

Facial recognition app designed using the library insightface with the model 'arcface_r100_v1'. It has been designed to provide recognition of faces within a defined group of people with a single photo of each person.



This app works on the follwing steps:
  Facial detection: uses 'retinaface_r50_v1' based facial detection to detect each face.
  Facial embedding extraction: It uses arcface_r100_v1 to extract embedding features from each people faces.
  Softmax classifier: It classifies the embedding vector using a softmax classifier that has been trained with a group of previously known and trained faces to check if the detected person is in the database.
  
  
This system was designed to work live with training and classifying and it gives the possibility to include new people in the group by extracting their embedding features and training the softmax classifier with it.



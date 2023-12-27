# Split_DNN

After dividing the VGG16 model by each layer, interence is performed separately on the edge server and client.

- edge_server_main.py: code for inference on an edge server
- client_vgg_main_test.py: code for inference on an edge device 
- communication.py: code for communication between the edge server and the edge device
- Golden_Retriever_Hund_Dog.jpg: one image file to perform inference on
- imagenet_class_index.json: the indexes of imagenet
- models/vgg16.py: pretrained VGG16 model


=how to run=

python3 edge_server_main.py

python3 client_vgg_main.py

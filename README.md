# VoiceConversion
This is a one-shot voice conversion model trained on [VCTK Dataset](https://paperswithcode.com/dataset/vctk).
To train the model, one needs to download the dataset and change the paths in the code to point towards it. Those should've probably been put in a config file but I can't be bothered to refactor this code now because this was a project I did as a student years ago. 

To train:
python src/train.py

To convert voice:

1)compute embeddings: python src/embedder/save_embeddings.py

2)python convert_voice.py <source_wav> <target_wav> <output_path>

# VoiceConversion
This is a one-shot voice conversion model.

To train:
python src/train.py

To convert voice:

1)compute embeddings: python src/embedder/save_embeddings.py

2)python convert_voice.py <source_wav> <target_wav> <output_path>

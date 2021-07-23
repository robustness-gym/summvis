python -m spacy download en_core_web_sm &&
mkdir -p data &&
mkdir -p preprocessing &&
curl https://storage.googleapis.com/sfr-summvis-data-research/cnn_dailymail_1000.validation.anonymized.zip > preprocessing/cnn_dailymail_1000.validation.anonymized.zip &&
unzip -o preprocessing/cnn_dailymail_1000.validation.anonymized.zip -d preprocessing/ &&
python preprocessing.py \
--deanonymize \
--dataset_rg preprocessing/cnn_dailymail_1000.validation.anonymized \
--dataset cnn_dailymail \
--version 3.0.0 \
--split validation \
--processed_dataset_path data/cnn_dailymail_10.validation \
--n_samples 10
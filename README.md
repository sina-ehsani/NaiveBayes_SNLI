Docker file:
`docker pull sinaehsani/deeplearning_essentials:latest`

Singularity pull:
`singularity pull shub://sinaehsani6/TextProject`

Download data:
```
wget https://nlp.stanford.edu/projects/snli/snli_1.0.zip` 
unzip -a snli_1.0.zip
rm -f snli_1.0.zip
```

To run preprocssing: (you can also download all the preprocessed data from this [Google Drive](https://drive.google.com/open?id=1xEvVzC2MG-WigYSGT4syaP-3shVXI2xF "Preprocessed Data") )
```
python prepro.py
```

To test the program run:
```
python project.py test
```

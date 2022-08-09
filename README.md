# Corpus Specific FastText word embeddings for Urdu Toxic Comment Detection

With the advent of various social media sites over the last decade, the popularity of user generated material on social media has surged, leading to an increase in toxic or hostile text on these sites. The Toxic texts have unpleasant effects on users of the platform and on community as a whole. This uncontrolled spread has surfaced as an undesirable social issue globally causing mental health problems and even suicides. As a result, recognising toxic comments is becoming increasingly important. While toxic comment detection is being extensively researched for languages that have abundant resources like English, there has been little to no research is being done for the Urdu language, that is scarce in resource yet widely spoken in South Asia. Urdu is spoken as a primary language by approximately 70 million people and as a second language by more than 100 million people [1], predominantly in Pakistan and India. Our work tackles the problem of Toxicity detection in Urdu language comment detection by generating an extensive and diverse labeled data for detecting toxicity and non-toxicity in comments in the Urdu language. The UTC (Urdu Toxic Comments) curated corpus comprised of 72 thousand comments transliterated from the RUT (Roman Urdu Toxicity) dataset contributed by [2]. Using this dataset Urdu harmful comments, we trained a variety of classification models based on Deep Learning and classic Machine Learning for detecting toxicity in the Urdu language comments, using TF-IDF vectors and cutting-edge deep learning based models relying on word embeddings. Despite the algorithms’ success in categorizing damaging statements in English, the lack of pre-trained embeddings for Urdu led to the creation of corpus-specific FastText word embeddings. Finally, we show that the CNN Tweaked model with corpus-specific FastText embeddings performs the best, with an F1-score of 0.8992, setting a new standard for hazardous comment identification in Urdu.

<img src="/images/comparison.jpg alt="Alt text" title="Experiment Comparison">

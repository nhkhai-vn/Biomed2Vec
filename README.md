# Biomed2Vec

Word2vec embeddings for biomedical concepts, including multi-word ones. Every UMLS concept name is detected in a 15.4-billion-token corpus (Wikipedia, PubMed, PMC) and joined into one token, so `hair_loss` gets its own vector next to the single words. With a minimum frequency of 10, the merged corpus yields vectors for over 650,000 concepts.

Research code from 2018 at the National Institute of Informatics, kept as a record; it is not maintained.

## Corpus

As of April 2018:

| Corpus | Tokens | Distinct biomedical concepts |
|---|---|---|
| Wikipedia | 2,374,659,439 | 655,187 |
| PubMed | 3,125,953,041 | 979,601 |
| PMC | 9,895,775,276 | 990,897 |
| Merged | 15,396,387,756 | 1,359,860 |

## Using a model

The models are saved in word2vec binary format, for example `pub_wikipedia_500.bin` (merged corpus, 500 dimensions, 1.22 GB):

```python
from gensim.models import KeyedVectors

vectors = KeyedVectors.load_word2vec_format("pub_wikipedia_500.bin", binary=True)
vectors.most_similar("hair_loss")
```

The pre-trained models and the UMLS resource files were shared from Google Drive; those links no longer work, so the models are not available for now.

## Training

The scripts in `src/` read and write `corpus/`, `model/`, `input/` and `tmp/` next to `src/`. They use the gensim 3 API (`Word2Vec(size=...)`, `wv.vocab`), so run them with Python 3.8 or older, `gensim<4` and `nltk`.

1. `download.sh` fetches the Wikipedia dump and the PubMed and PMC bulk files. The PMC paths it uses (`oa_bulk`, `manuscript`, `historical_ocr`) have since moved; take the current ones from PMC's bulk download page.
2. `corpus.py` extracts the text and tokenizes it into `corpus/tokenized/`.
3. `detector.py detect <input> <output>` joins every UMLS concept name (from `concepts.txt`, up to 15 tokens) into one token. `detector.py count <corpus> <output>` counts concept frequencies.
4. `wordvec.py train <corpus> <dimensions> [epochs]` trains CBOW word2vec (window 5, minimum count 10) and saves `model/<corpus>_<dimensions>.bin`.
5. `wordvec.py benchmark <shared_vocab> <models...>` and `bench.py <model>` compare models on UMLS synonyms: for one name of each concept, the rank at which another name with the same CUI first appears among its nearest neighbours (lower is better).

The UMLS files (`concepts.txt`, `umls_concepts.txt`, `cui_pstr`) are not included; they come from a licensed UMLS release.

## License

[Apache-2.0](LICENSE)

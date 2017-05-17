
# Word Sense Disambiguation Knowledge Sources
To train an all-word WSD model, several knowledge sources are neccesary. Including :
- Sense inventory
- Sense labeled corpus
- Embeddings

Here we explore how to leverage these knowledge sources using Python.

## Sense inventory word-net

The powerful NLTK package includes handy wordnet interface. The senses of a word is stored as form of synset, which includes the definition of a sense and words related to this sense.

For a more detailed documentation of wordnet interface, please refer to : http://www.nltk.org/howto/wordnet.html


```python
from nltk.corpus import wordnet as wn
```

To get access to all the synsets of a particular word, use wn.synsets(word).


```python
# Example. 
wn.synsets('qualify')
```




    [Synset('qualify.v.01'),
     Synset('qualify.v.02'),
     Synset('qualify.v.03'),
     Synset('qualify.v.04'),
     Synset('stipulate.v.01'),
     Synset('qualify.v.06'),
     Synset('modify.v.02')]



Under every synnet, it contains the lemmas that are  in this synnet. use .lemmas() to access them.


```python
# Example: Let's choose the third synnet of the word 'qualify', which is not its usual sense.
qualify_03 = wn.synsets('qualify')[2]
# It is also easy to print out its definition
print('qualify_.v.03 definition: %s'%(qualify_03.definition()))
qualify_03.lemmas()
```

    qualify_.v.03 definition: make more specific





    [Lemma('qualify.v.03.qualify'), Lemma('qualify.v.03.restrict')]



## Sense labeled corpus
NLTK also contains wordnet's senses labeled corpus: SemCor. For a detail documentation of NTLK tagged corpora please refer to:
http://www.nltk.org/howto/corpus.html#tagged-corpora


```python
from nltk.corpus import semcor
```

SemCor corpus is a chunk corpus that consists of tree. The senses taged are associated with each tree.


```python
example_chunk = semcor.tagged_sents(tag='sem')[0]
example_chunk
```




    [['The'],
     Tree(Lemma('group.n.01.group'), [Tree('NE', ['Fulton', 'County', 'Grand', 'Jury'])]),
     Tree(Lemma('state.v.01.say'), ['said']),
     Tree(Lemma('friday.n.01.Friday'), ['Friday']),
     ['an'],
     Tree(Lemma('probe.n.01.investigation'), ['investigation']),
     ['of'],
     Tree(Lemma('atlanta.n.01.Atlanta'), ['Atlanta']),
     ["'s"],
     Tree(Lemma('late.s.03.recent'), ['recent']),
     Tree(Lemma('primary.n.01.primary_election'), ['primary', 'election']),
     Tree(Lemma('produce.v.04.produce'), ['produced']),
     ['``'],
     ['no'],
     Tree(Lemma('evidence.n.01.evidence'), ['evidence']),
     ["''"],
     ['that'],
     ['any'],
     Tree(Lemma('abnormality.n.04.irregularity'), ['irregularities']),
     Tree(Lemma('happen.v.01.take_place'), ['took', 'place']),
     ['.']]



To access the sense of each tree, we need to use the .synset() method of lemma object to access its belonged synset.


```python
example_tree = example_chunk[1]
example_tree.label().synset()
```




    Synset('group.n.01')



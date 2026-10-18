"""Historical K=9 LDA notebook branch, adapted to package paths.

This extract retains the preprocessing and alpha sweep of notebook cells
9, 11-15, 18, 21, 24, 32, 34 and 61-67. Exploratory LSI/MALLET branches
are omitted. It is not called by run_pipeline.py. A new fit is not
assumed to recover the saved document assignments or topic geometry.
"""
from pathlib import Path
import os
import gensim
from gensim.utils import simple_preprocess
import numpy as np
import pandas as pd
import nltk
import tqdm

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUTPUT = ROOT / "output" / "lda_new_fit"
OUTPUT.mkdir(parents=True, exist_ok=True)
for name in ["summary", "figures", "weights", "keywords", "documents"]:
    (OUTPUT / name).mkdir(exist_ok=True)
os.chdir(OUTPUT)

df = pd.read_csv(DATA / 'candidate_posts.csv')
df['translation'] = df['translation'].apply(str)
df['translation'] = pd.DataFrame(df, columns = ['translation'])


import nltk
stop_words = nltk.corpus.stopwords.words('english')
stopdata=pd.read_csv(DATA / 'lda_custom_stopwords.csv')
stop_words=stop_words+stopdata['word'].tolist()


def normalize_corpus(texts):
    norm_texts = []
    for text in tqdm.tqdm(texts):
        text_tokens = gensim.utils.simple_preprocess(str(text), deacc=True)
        text_tokens = [token for token in simple_preprocess(str(text)) if token not in stop_words and len(token) >= 2]              
        norm_texts.append(text_tokens)
    return norm_texts

norm_df = normalize_corpus(df['translation'])

import spacy
import tqdm
nlp = spacy.load("en_core_web_md")
                                   

import en_core_web_md
nlp = en_core_web_md.load(disable=['parser', 'ner'])

                                                       
def lemmatization(texts, allowed_postags=['NOUN', 'VERB']):
    texts_out = []
    for sent in tqdm.tqdm(texts):
        doc = nlp(" ".join(sent))
        texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
    return texts_out

                                                                                            
norm_df_lemmatized_NVA = lemmatization(norm_df,  allowed_postags=['NOUN', 'VERB', 'ADJ'])            
                                                                          

bigram = gensim.models.Phrases(norm_df_lemmatized_NVA, min_count=20, threshold=20)                                                  
bigram_model = gensim.models.phrases.Phraser(bigram)
norm_corpus_bigrams = [bigram_model[doc] for doc in norm_df_lemmatized_NVA]

stop_words.extend(['do','make','go','come','much','from','subject','feel','turn','set','say','right','left','camping','camp','park','ctrip',"locate",
                   're', 'edu', 'use', 'amp', 's', 'be', 'need','small','large','come','look','simple',"next","allow",'camp','camping','willing',
                   'previous','take','want','know','shazu','become','first','second','whole','next','full','include',
                   'mean','right','back','get','find','see','guy','open','base','big','boss','easy','start','spring','winter','autumn','summer',
                   'seem','many','less','lot','little','sure','look','queue','point','half','recommend','recommendable','regret','red','return','difficult','main','site','low','general',
                   'tell','way','ask','keep','doordash','postmate','man','original','opportunity',
                   'ubereats','caviar','grubhub','door dash','ubereat','possibility','word','work','long','slope','last','love',
                   'door_dash','airbnb','post_mate','uber','grub hub','thank','know','zhuoer','well','good','great','nice','bad'])

def remove_stopwords(texts):
    return [[word for word in simple_preprocess(str(doc)) if word not in stop_words and len(word) >= 3] for doc in texts]

norm_corpus_final = remove_stopwords(norm_corpus_bigrams)

df['lemmatized'] = norm_corpus_final


                                                          
dictionary = gensim.corpora.Dictionary(norm_corpus_final)
print('Sample word to number mappings:', list(dictionary.items())[:15])
print('Total Vocabulary Size:', len(dictionary))


                                                                                       
dictionary.filter_extremes(no_below=2, no_above=0.2)             
print('Total Vocabulary Size:', len(dictionary))

                                               
bow_corpus = [dictionary.doc2bow(text) for text in norm_corpus_final]
print([(dictionary[idx] , freq) for idx, freq in bow_corpus[0][:50]])               
print('Total number of documents:', len(bow_corpus))

alpha = list(np.arange(0.01, 1, 0.3))
alpha.append('symmetric')
alpha.append('asymmetric')

     
def basic_modle(k,a):
  TOTAL_TOPICS = k
  lda_model = gensim.models.LdaModel(corpus=bow_corpus, id2word=dictionary, chunksize=400,
                                   alpha=a, eta='auto', random_state=100,                               
                                   iterations=500, num_topics=TOTAL_TOPICS,
                                   passes=20, eval_every=None)                          
  return lda_model

      
def cvandumass(lda_model):
  cv_coherence_model_lda = gensim.models.CoherenceModel(model=lda_model, corpus=bow_corpus,
                                                        texts=norm_corpus_final,
                                                        dictionary=dictionary,
                                                        coherence='c_v')
  avg_coherence_cv = cv_coherence_model_lda.get_coherence()

  umass_coherence_model_lda = gensim.models.CoherenceModel(model=lda_model, corpus=bow_corpus,
                                                          texts=norm_corpus_final,
                                                          dictionary=dictionary,
                                                          coherence='u_mass')
  avg_coherence_umass = umass_coherence_model_lda.get_coherence()

  perplexity = lda_model.log_perplexity(bow_corpus)
  return avg_coherence_cv,avg_coherence_umass,perplexity


def topword(lda_model,num_words):
  topics = [[(term, round(wt, 3))
                for term, wt in lda_model.show_topic(n, topn=num_words)]
                    for n in range(0, lda_model.num_topics)]
  topics_df = pd.DataFrame([[term for term, wt in topic] for topic in topics],
                          columns = ['Term'+str(i) for i in range(1, num_words+1)],
                          index=['Topic '+str(t) for t in range(1, lda_model.num_topics+1)]).T

  topics_df_wt = pd.DataFrame([[(term,wt) for term, wt in topic] for topic in topics],
                          columns = ['Term'+str(i) for i in range(1, num_words+1)],
                          index=['Topic '+str(t) for t in range(1, lda_model.num_topics+1)]).T
  return topics_df_wt


def topicsdf(lda_model):
  topics = [[(term, round(wt, 3))
               for term, wt in lda_model.show_topic(n, topn=30)]
                   for n in range(0, lda_model.num_topics)]
  topics_df = pd.DataFrame([', '.join([term for term, wt in topic])
                              for topic in topics],
                         columns = ['Terms per Topic'],
                         index=['Topic'+str(t) for t in range(1, lda_model.num_topics+1)]
                         )

  return topics_df

def topic2doc(lda_model,topics_df):
  tm_results = lda_model[bow_corpus]
  corpus_topics = [sorted(topics, key=lambda record: -record[1])[0]
                      for topics in tm_results]
  corpus_topic_df = pd.DataFrame()
  df = pd.read_csv(DATA / 'candidate_posts.csv')
  corpus_topic_df['Document'] = range(0, len(df['translation']))                
  corpus_topic_df['Dominant Topic'] = [item[0]+1 for item in corpus_topics]
  corpus_topic_df['Contribution %'] = [round(item[1]*100, 2) for item in corpus_topics]
  corpus_topic_df['Topic Desc'] = [topics_df.iloc[t[0]]['Terms per Topic'] for t in corpus_topics]
  corpus_topic_df['Text'] = df['translation']                
  return corpus_topic_df


import pyLDAvis
import pyLDAvis.gensim_models as gensimvis
import os
folder_path1 = "summary"
folder_path3 = "figures/"
folder_path2 = "weights"
folder_path4 = "keywords"
folder_path5 = "documents"                                                             



k=9                                                
model_results = {'Topics': [],
                 'Alpha': [],
                 'Beta': [],
                 'Coherence': [],
                 'Umass':[],
                 'perplexity':[]
                }
for a in alpha:
  lda_model=basic_modle(k=k,a=a)
  avg_coherence_cv,avg_coherence_umass,perplexity=cvandumass(lda_model)
  model_results['Topics'].append(k)
  model_results['Alpha'].append(a)
  model_results['Beta'].append('auto')
  model_results['Coherence'].append(avg_coherence_cv)
  model_results['Umass'].append(avg_coherence_umass)
  model_results['perplexity'].append(perplexity)
  topics_df_wt=topword(lda_model=lda_model,num_words=k)
  topics_df_wt.to_csv("0124"+str(lda_model.num_topics) +"topic_"+'alpha_%s.csv'%str(a),index=False)
  topics_df_wt.to_csv(os.path.join(folder_path2, "0124"+str(lda_model.num_topics) +"topic_"+'alpha_%s.csv'%str(a)), index=False)

  topics_df=topicsdf(lda_model=lda_model)
  topics_df.to_csv("0124keyword"+str(lda_model.num_topics) +"topic_"+'alpha_%s.csv'%str(a),index=False)
  topics_df.to_csv(os.path.join(folder_path4, "0124keyword"+str(lda_model.num_topics) +"topic_"+'alpha_%s.csv'%str(a)), index=False)

  corpus_topic_df=topic2doc(lda_model=lda_model,topics_df=topics_df)
  corpus_topic_df.to_csv("0124content"+str(lda_model.num_topics) +"topic_"+'alpha_%s.csv'%str(a),index=False)
  corpus_topic_df.to_csv(os.path.join(folder_path5, "0124content"+str(lda_model.num_topics) +"topic_"+'alpha_%s.csv'%str(a)), index=False)

  
  pyLDAvis_data = gensimvis.prepare(lda_model, bow_corpus, dictionary, sort_topics=False)
  pyLDAvis.save_html(pyLDAvis_data, "0124pic_"+str(lda_model.num_topics) +"topic_"+'alpha_%s.html'%str(a))
  pyLDAvis.save_html(pyLDAvis_data, folder_path3 + "0124pic_" + str(lda_model.num_topics) + "topic_" + 'alpha_%s.html' % str(a))

                                  


pd.DataFrame(model_results).to_csv("0124"+str(lda_model.num_topics) +"topic_"+"model_statistics.csv", index=False)
pd.DataFrame(model_results).to_csv(os.path.join(folder_path1, "0124"+str(lda_model.num_topics) +"topic_"+"model_statistics.csv"), index=False)


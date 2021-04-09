from transformers import BertTokenizer, BertForQuestionAnswering
import torch
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report
from nltk.tokenize import word_tokenize
import numpy as np
import random
import gensim
import os
import json

tokenizer = BertTokenizer.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')
model = BertForQuestionAnswering.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')
if torch.cuda.is_available():
  model.to('cuda')

def answer_question(question, text):
  print('question: ', question)
  # print('passage', text)

  input_ids = tokenizer.encode(question, text, padding=True, truncation=True, max_length=510, add_special_tokens = True)

  # Search the input_ids for the first instance of the `[SEP]` token.
  sep_index = input_ids.index(tokenizer.sep_token_id)

  # The number of segment A tokens includes the [SEP] token istelf.
  num_seg_a = sep_index + 1

  # The remainder are segment B.
  num_seg_b = len(input_ids) - num_seg_a

  # Construct the list of 0s and 1s.
  segment_ids = [0]*num_seg_a + [1]*num_seg_b

  # Ensure all input_id has been assigned segment_id.
  assert len(segment_ids) == len(input_ids)

  input_tensors = torch.tensor([input_ids]).to('cuda') if torch.cuda.is_available() else torch.tensor([input_ids])
  segment_id_tensors = torch.tensor([segment_ids]).to('cuda') if torch.cuda.is_available() else torch.tensor([segment_ids])

  # Run our example through the model.
  outputs = model(input_tensors,
                  token_type_ids=segment_id_tensors,
                  return_dict=True) 

  start_scores = outputs.start_logits
  end_scores = outputs.end_logits

  # Find the tokens with the highest `start` and `end` scores.
  answer_start = torch.argmax(start_scores)
  answer_end = torch.argmax(end_scores)

  # Get the string versions of the input tokens.
  tokens = tokenizer.convert_ids_to_tokens(input_ids)

  # Start with the first token.
  answer = tokens[answer_start]

  # Select the remaining answer tokens and join them with whitespace.
  for i in range(answer_start + 1, answer_end + 1):

      # If it's a subword token, then recombine it with the previous token.
      if tokens[i][0:2] == '##':
          answer += tokens[i][2:]

      # Otherwise, add a space then the token.
      else:
          answer += ' ' + tokens[i]

  return answer

def similarity_model(ans_str):

  tokenized_docs = list(map(lambda ans: [w.lower() for w in word_tokenize(ans)], ans_str))

  dictionary = gensim.corpora.Dictionary(tokenized_docs)

  corpus = [dictionary.doc2bow(doc) for doc in tokenized_docs]

  tf_idf = gensim.models.TfidfModel(corpus)

  sims = gensim.similarities.docsim.MatrixSimilarity(tf_idf[corpus],num_features=len(dictionary))

  return (dictionary, sims, tf_idf)

def cos_sim(dictionary, sim_model, tf_idf, ans_str):

  query_doc = [w.lower() for w in word_tokenize(ans_str)]

  query_doc_bow = dictionary.doc2bow(query_doc)

  # perform a similarity query against the corpus
  query_doc_tf_idf = tf_idf[query_doc_bow]

  probability = sim_model[query_doc_tf_idf]

  return probability

def pick_answer_from_probabilities(probabilities, choice_count = 4):
  option_probabilities =  {'A':'','B':'','C':''} if choice_count == 3 else  {'A':'','B':'','C':'', 'D': ''} 

  for opt, proba in zip(option_probabilities.keys(), probabilities):
    option_probabilities[opt] = proba

  print('Probabilities for this question:', option_probabilities)

  max_probability = max(option_probabilities.values())
  if max_probability == 0:
      random_picked_answer = random.choice(['A','B','C'] if choice_count == 3 else ['A','B','C'] )
      print('Random picked option: {}'.format(random_picked_answer))
      return random_picked_answer
  else:
    for key, val in option_probabilities.items(): 
      if val == max_probability:
        print('Predicted option: {}'.format(key))
        return key

def readRACE(relativeDocPath):
    dataset = []

    for filename in os.listdir(os.path.join(os.getcwd(), relativeDocPath)):
        with open(os.path.join(relativeDocPath, filename), 'r') as f: # open in readonly mode
            story = json.load(f)
        
        temp = {}
        temp["article"] = story["article"].replace('\\n', ' ')
        # print(temp["article"])
        tempPerQuestion = {}
        for i in range(len(story["questions"])):
            temp = {"article": temp["article"]}   

            temp["question"] = story["questions"][i]
            for j in range(len(story["options"][i])):
                temp[f"choice {j}"] = story["options"][i][j]
                
            # answer choice = A/B/C/D, answer index = 0/1/2/3, answer = answer in string format
            temp["answer index"] = ord(story["answers"][i]) - 65      # from "A" to 0
            temp["answer"] = temp[f"choice {temp['answer index']}"]
            temp["answer choice"] = story["answers"][i]

            dataset.append(temp)

    return pd.DataFrame(dataset)

if __name__ == "__main__":
  
  race_data = readRACE('./data/RACE/test/middle')

  all_predicted_options = []
  for i in range(len(race_data)):
    predicted_answer_string = answer_question(race_data['question'][i], race_data['article'][i])
    print('Predicted answer string: "{}"'.format(predicted_answer_string))

    correct_ans = race_data['answer choice'][i]
    print('Correct option is '+ correct_ans)

    option_strings = list(map(lambda choice: race_data.loc[i][choice], ['choice 0', 'choice 1', 'choice 2', 'choice 3']))
    options_map = {'A': option_strings[0], 'B': option_strings[1], 'C': option_strings[2], 'D': option_strings[3] }

    dictionary, sims, tf_idf = similarity_model(option_strings)

    probabilities = cos_sim(dictionary, sims, tf_idf, predicted_answer_string)
    
    picked_answer = pick_answer_from_probabilities(probabilities)
    print('Answer precited: "{}"\n'.format(options_map[picked_answer]))
    
    all_predicted_options.append(picked_answer)


  assert len(all_predicted_options) == len(race_data)

  print('Accuracy', accuracy_score(race_data['answer choice'], all_predicted_options))
  print(classification_report(race_data['answer choice'], all_predicted_options))
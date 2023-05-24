# Quest4

아이펠캠퍼스 온라인4기 피어코드리뷰
.
- 코더 : 임지혜
- 리뷰어 : 김동규
-------------------------------------------------- -----------

PRT(PeerReviewTemplate)

- [?] 코드가 정상적으로 동작하고 주어진 문제를 해결했나요?

목표한 정확도 달성
```python
accuracy: 0.8617
```

미리 학습된 한국어 워드 임베딩 사용하여 서로 비교함
```python
#사전학습된 워드임베딩 모델: Word2Vec활용
word2vec_file_path = "/gdrive/MyDrive/AIFFEL/230522/data/word2vec_ko.model"
word_vectors = Word2VecKeyedVectors.load(word2vec_file_path)
vector = word_vectors.wv["재미"]
len(vector)

word_vectors.wv.similar_by_word("재미") 
...

[('묘미', 0.6163142919540405),
 ('취미', 0.6034970283508301),
 ('흥미', 0.5939850211143494),
 ('유머', 0.5888698101043701),
 ('보람', 0.5689517259597778),
 ('즐거움', 0.5631207823753357),
 ('개그', 0.5552946329116821),
 ('이야기', 0.5536993741989136),
 ('연애', 0.552293598651886),
 ('열의', 0.546456515789032)]
```

일부 모델에 대한 학습 결과가 없음.

- [O] 주석을 보고 작성자의 코드가 이해되었나요?

자칫하면 알아보기 힘들 전처리 부분에 주석을 달아서 무슨 과정을 수행했는지 빠르게 파악됨
```python
#데이터 전처리 및 word_to_index 생성
tokenizer = Mecab()
stopwords = ['의','가','이','은','들','는','좀','잘','걍','과','도','를','으로','자','에','와','한','하다']
num_words = 10000

def load_data(train_data, test_data, num_words = num_words):
  #데이터 중복제거
  train_data.drop_duplicates(subset=['document'], inplace=True)
  test_data.drop_duplicates(subset=['document'], inplace=True)
  
  #NaN 결측치 제거
  train_data = train_data.dropna(how = 'any')   
  test_data = test_data.dropna(how = 'any') 

  #토큰화 및 불용어 제거
  X_train = []
  for sentence in train_data['document']:
    temp_X = tokenizer.morphs(sentence)   # 한국어 토크나이저로 토큰화
    temp_X = [word for word in temp_X if not word in stopwords]  # 불용어(Stopwords) 제거
    X_train.append(temp_X)
```

주석을 통해 무엇을 하려고 했는지 파악 쉬움
```python
#데이터셋 내 문장 길이 분포
total_data_text = list(X_train) + list(X_test)

num_tokens = [len(tokens) for tokens in total_data_text]
num_tokens = np.array(num_tokens)
print('문장길이 평균 : ', np.mean(num_tokens))
print('문장길이 최대 : ', np.max(num_tokens))
print('문장길이 표준편차 : ', np.std(num_tokens))

#text 길이 분포 -> 20이후부터 급격하게 감소, 60이후로 데이터 거의 없음 
plt.hist([len(s) for s in total_data_text], bins=50)
plt.show()
     
```

- ['X] 코드가 에러를 유발할 가능성이 있나요?

파일 사용시 잊지않고 close함
```python
f = open(word2vec_file_path, 'w')
f.write('{} {}\n'.format(vocab_size-4, word_vector_dim))

#단어개수만큼의 워드벡터 파일에 기록
vectors = model.get_weights()[0]
for i in range(4, vocab_size):
  f.write('{} {}\n'.format(index_to_word[i], ' '.join(map(str, list(vectors[i, :])))))
f.close()
```

절대 경로를 사용하여 파이썬 인터프리터 상의 CWD의 영향을 받지 않음
```python
word2vec_file_path = "/gdrive/MyDrive/AIFFEL/230522/data/word2vec_ko.model"

```

- [O] 코드 작성자가 코드를 제대로 이해하고 작성했나요? (직접 인터뷰해보기)

"""
loss를 그려 보면, 몇 epoch까지의 트레이닝이 적절한지 최적점을 추정가능
validation loss의 그래프가 train loss와의 이격이 발생하게 되면 
더 이상의 트레이닝은 무의미
"""
라고 적었는데 이 역할을 해주는 텐서플로의 기능은 무엇인가.
답: EarlyStopping이라고 들어본거 같다.

노드 예제 속 3개의 모델은 종종 과적합이 일어나는 경향이 있는데 이를 방지할 수 있는 수단은 무엇이 있을까
답: Dropout, Test 데이터와 Train 데이터 조절 등

- [O] 코드가 간결한가요?

plot쪽에서 코드가 길고, 모델 생성쪽도 라인수가 길지만, 불필요한 중복 코드는 없는 것 같음
```python
plt.subplot(1,2,1)
plt.plot(epochs, loss, 'bo', label='Training loss') # "bo"는 "파란색 점"입니다
plt.plot(epochs, val_loss, 'b', label='Validation loss') # b는 "파란 실선"입니다
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
```

----------------------------------------------

참고 링크 및 코드 개선
- Early stopping: https://keras.io/api/callbacks/early_stopping/
- Callback 적용: https://jins-sw.tistory.com/27

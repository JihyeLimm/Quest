# Quest8

아이펠캠퍼스 온라인4기 피어코드리뷰
.
- 코더 : 임지혜
- 리뷰어 : 정연준
-------------------------------------------------------------

PRT(PeerReviewTemplate)

- [o] 코드가 정상적으로 동작하고 주어진 문제를 해결했나요?

![image](https://github.com/JihyeLimm/Quest/assets/131635437/6a1c420c-5b73-4bc4-8618-738ee3657aa6)

학습의 결과물로 한국어 입력에 따라 한국어 출력값을 얻었다.

- [O] 주석을 보고 작성자의 코드가 이해되었나요?

- 주석과 함께 작성자와 인터뷰를 함께하여 전반적으로 코드의 내용이 이해되었다.

- [X] 코드가 에러를 유발할 가능성이 있나요?

```python
class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):

  def __init__(self, d_model, warmup_steps=4000):
    super(CustomSchedule, self).__init__()
    self.d_model = d_model
    self.d_model = tf.cast(self.d_model, tf.float32)
    self.warmup_steps = warmup_steps

  def __call__(self, step):
    step = tf.cast(step, tf.float32)   #optimizer선언할때 에러나서 추가 
    arg1 = tf.math.rsqrt(step)
    arg2 = step * (self.warmup_steps**-1.5)

    return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)
```
- colab에서 call 부분에 
```python
step = tf.cast(step, tf.float32)
```
부분을 추가하지 않으면 에러가 발생할 수 있다고 한다.

- 텐서플로우 버전에 다른 경우 step의 타입을 설정해주어 이를 방지하였다.


- [o] 코드 작성자가 코드를 제대로 이해하고 작성했나요? (직접 인터뷰해보기)

![image](https://github.com/JihyeLimm/Quest/assets/131635437/d4122d48-a0aa-4f14-9044-eb55849192a9)

```python
#단어 정수인코딩 & 최대길이 초과샘플 제거 & 패딩
MAX_LENGTH = 12

def tokenize_and_filter(inputs, outputs):
  tokenized_inputs, tokenized_outputs = [], []
  
  for (sentence1, sentence2) in zip(inputs, outputs):
    # 정수 인코딩 과정에서 시작 토큰과 종료 토큰을 추가
    sentence1 = START_TOKEN + tokenizer.encode(sentence1) + END_TOKEN
    sentence2 = START_TOKEN + tokenizer.encode(sentence2) + END_TOKEN

    # 최대 길이 12 이하인 경우에만 데이터셋으로 허용
    if len(sentence1) <= MAX_LENGTH and len(sentence2) <= MAX_LENGTH:
      tokenized_inputs.append(sentence1)
      tokenized_outputs.append(sentence2)
  
  # 최대 길이 12 로 모든 데이터셋을 패딩
  tokenized_inputs = tf.keras.preprocessing.sequence.pad_sequences(
      tokenized_inputs, maxlen=MAX_LENGTH, padding='post')
  tokenized_outputs = tf.keras.preprocessing.sequence.pad_sequences(
      tokenized_outputs, maxlen=MAX_LENGTH, padding='post')
  
  return tokenized_inputs, tokenized_outputs


questions, answers = tokenize_and_filter(questions, answers)
print('단어장의 크기 :',(VOCAB_SIZE))
print('필터링 후의 질문 샘플 개수: {}'.format(len(questions)))   #10744
print('필터링 후의 답변 샘플 개수: {}'.format(len(answers)))     #10744
```

- 데이터 샘플의 길이 분포를 시각화한 다음 이를 바탕으로 적절한 샘플크기를 설정하여 패딩하였다.

- [O] 코드가 간결한가요?

- 더 간결하게 할 수 있는 여지가 보이지 않습니다.

----------------------------------------------

참고 링크 및 코드 개선

- 코드 리뷰 시 참고한 링크가 있다면 링크와 간략한 설명을 첨부합니다.
- 코드 리뷰를 통해 개선한 코드가 있다면 코드와 간략한 설명을 첨부합니다.

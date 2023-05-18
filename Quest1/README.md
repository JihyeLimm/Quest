# Quest1

아이펠캠퍼스 온라인4기 피어코드리뷰
.
- 코더 : 임지혜
- 리뷰어 : 사재원
-------------------------------------------------- -----------

PRT(PeerReviewTemplate)

- [O] 코드가 정상적으로 동작하고 주어진 문제를 해결했나요?
```
네 데이터 불러오는 부분만 알맞게 설정해준다면 정상적으로 동작하면서 주어진 조건을 만족시켰습니다.
```

- [O] 주석을 보고 작성자의 코드가 이해되었나요?
  
  네 잘 이해가되었습니다 코드 부분마다 주석이 달려있어 이해하기쉬웠습니다

- [O] 코드가 에러를 유발할 가능성이 있나요?
  
  코드를 응용하게될 때 데이터 불러오는 위치만 잘 적용 해준다면 문제없을거같습니다.

- [O] 코드 작성자가 코드를 제대로 이해하고 작성했나요? (직접 인터뷰해보기)
  
  오차 계산 공식도 맞고 직접물어보니 개념에 대해서도 잘 이해하신것같습니다.
  
  시각화 방법또한 축의 라벨도 지정하면서 가독성 또한 좋아 활용방식도 잘 터득한 것 같습니다.

- [O] 코드가 간결한가요?
  
  1번 문제와 같이 각 단계별로 함수 형태로 만들고 가독성이 좋게 만들었습니다.
```
def model(X, W, b):
    predictions = 0
    for i in range(10):
        predictions += X[:, i] * W[i]
    predictions += b
    return predictions

#손실함수loss정의
def MSE(a, b):
    mse = ((a - b) ** 2).mean()  # 두 값의 차이의 제곱의 평균
    return mse

def loss(X, W, b, y):
    predictions = model(X, W, b)
    L = MSE(predictions, y)
    return L
```

또한 변수 네이밍또한 언더바를 활용하여 어떤역할인지 확실하게 알 수 있어 간결하면서 보기좋았습니다.
```
LEARNING_RATE = 0.1
```

이런 부분 같은 경우는 사용자마다 느끼는 바가 다를수도있고 저도 고쳐야 할점 이지만 
하나로 묶어서 하는 방식이 존재하여 만약 코드를 좀 더 간결하게 작성하고싶다면 묶는방법도 좋은거 같습니다
```
#연, 월, 일, 시, 분, 초까지 6가지 컬럼 생성
train['year']=train[ "datetime"].dt.year
train['month']=train[ "datetime"].dt.month
train['day']=train[ "datetime"].dt.day
train['hr']=train[ "datetime"].dt.hour
train['min']=train[ "datetime"].dt.minute
train['sec']=train[ "datetime"].dt.second
train.tail(3)
```

----------------------------------------------

참고 링크 및 코드 개선
- 코드 리뷰 시 참고한 링크가 있다면 링크와 간략한 설명을 첨부합니다.
- 코드 리뷰를 통해 개선한 코드가 있다면 코드와 간략한 설명을 첨부합니다.

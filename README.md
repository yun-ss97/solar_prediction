# 태양광 발전량 예측모델

## 주제

예시로 제시된 지역의 기상 데이터와 과거 발전량 데이터를 활용하여, 시간대별 태양광 발전량을 예측

* 기대효과: 시간대별 소비자 그룹의 전력소비량 예측 데이터와 결합하여 가장 효율적인 시간대별 태양광 발전과 국가 전력망을 조합 가능. 각 소비자 그룹에 최적화된 공급계획 수립 가능.  



## task 설명

태양광 발전은 매일의 기상 상황과 계절에 따른 일사량의 영향을 받는다. 이에 대한 예측이 가능하다면 보다 원활하게 전력 수급 계획을 세우는 것이 가능하다
 

신재생에너지의 생산 효율성을 극대화하고, 사용자들에게 저렴한 전력을 공급할 수 있도록 인공지능 기반 태양광 발전량 예측 모델을 구축하는 task이다. 

모델은 7일(Day 0~ Day6) 동안의 데이터를 인풋으로 활용하여, 향후 2일(Day7 ~ Day8) 동안의 30분 간격의 발전량(TARGET)을 예측해야 한다. 

(1일당 48개씩 총 96개 타임스텝에 대한 예측)


## 구축한 예측모델

1. *LightGBM, XGBoost*: boosting 계열의 tree 기반 ensemble model (데이터 iid 가정)

2. *1D CNN*: 1D Convolution 연산을 통해 48~96개 데이터씩 sliding 하면서 feature를 추출함

3. *Neural Network*: linear layer로 구성된 neural network 기반의 모델


## 데이터 source: [[데이콘] 태양광 발전량 예측 AI 경진대회](https://dacon.io/competitions/official/235680/overview/description/)


## 변수 설명
* 기본 변수

|변수|설명|
|:---:|:---:|
|Hour|시간|
|Minute|분|
|DHI (Diffuse Horizontal Irradiance($W/m^2$))| 태양광선이 대기 통과하는 동안 산란되어 도달하는 햇볕|
|DNI (Direct Normal Irradiance($W/m^2$))|태양으로부터 지표에 직접 도달하는 햇볕|
|WS (Wind Speed($m/s$))|풍속|
|RH(Relative Humidity(%))|상대습도|
|T (Temperature ($&deg;C$))|기온|
|Target ($Kw$)| 태양광 발전량|

* 파생변수
|변수|설명|
|:---:|:---:|
|GHI (Global Horizontal Irradiance ($W/m^2$))| 전체 도달 태양에너지 량|
|theta | 태양과 지면 간의 각도|



## Loss Function: Pinball Loss (=Quantile Loss)

모델별로 Pinball loss를 정의하는 방식에 차이가 있음.

1) XGBoost
* loss 정의
```python
def gradient(predt: np.ndarray, dtrain: xgb.DMatrix) -> np.ndarray:
    y = dtrain.get_label()
    errors = y - predt
    left_mask = errors >= 0
    right_mask = errors < 0
    return -quantile * left_mask + (1 - quantile) * right_mask

def hessian(predt: np.ndarray, dtrain: xgb.DMatrix) -> np.ndarray:
    y = dtrain.get_label()
    return np.ones_like(predt)

def quantile_loss(predt: np.ndarray,
                dtrain: xgb.DMatrix) -> Tuple[np.ndarray, np.ndarray]:
    grad = gradient(predt, dtrain)
    hess = hessian(predt, dtrain)
    return grad, hess
```

* model 생성
```python
model = xgb.train({'tree_method': 'gpu_hist'},
                      dtrain=dtrain,
                      obj = quantile_loss, num_boost_round = 100000,
                      evals= watch_list,
                      early_stopping_rounds = 50)
```

2) LightGBM
* Quantile Loss가 lightgbm 라이브러리 내부에 내장되어 있음.

* model 생성
```python
model = LGBMRegressor(objective='quantile', alpha=q, max_depth=128, boosting='gbdt',
                      n_estimators=750, num_leaves=152, bagging_fraction=0.5, learning_rate=0.02)                   
    
model.fit(X_train, Y_train, eval_metric = ['quantile'], 
          eval_set=[(X_valid, Y_valid)], early_stopping_rounds=512, verbose=500)
```

3) 1D CNN, Neural Network
* loss 정의
```python
def quantile_loss(pred, gt, quantile):
    qs = quantile
    sum_loss = 0
    loss = gt - pred
    loss = torch.max(qs*loss, (qs-1)*loss)
    sum_loss = torch.mean(loss)
    return sum_loss
```

* loss 사용
```python
loss = quantile_loss(output, target, quantile)
```
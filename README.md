🏆 Personalized Medicine: Redefining Cancer Treatment
===
### *Kaggle☁ Predict the effect of Genetic Variants to enable Personalized Medicine!😀*    
                                                                                                                   
***

## 1. Dataset
* 데이터셋의 전체 크기는 총 3316으로 학습 데이터셋으로는 부족한 상태입니다.
* 하나의 문서당 최대 십만 글자까지 들어있기 때문에 최대 토큰수가 512인 BERT를 사용하기에는 무리가 있습니다.
* 따라서 **한 텍스트를 2000글자(대략 512토큰 이내)로 쪼개 새롭게 문서를 만들어** Data augmentation과 BERT max token length 문제를 동시에 해결했습니다.
* 실제로 실험을 진행해보니 data augmentation을 진행한 데이터셋을 학습시킨 모델이 그렇지 않은 모델보다 더 성능이 좋은 것을 확인할 수 있었습니다.    
![image](https://user-images.githubusercontent.com/74829786/177870405-2029e627-8adc-470a-bccd-7a7d8be5223b.png)

## 2. Model
* `BioBERT-Large v1.1`를 사용하였습니다.
* ID Embedding을 적용한 모델과 적용하지 않은 모델을 설계하여 실험을 진행하였습니다.

## 3. Experiments
* ID Embedding을 적용한 경우    
![image](https://user-images.githubusercontent.com/74829786/177868532-eb173fd2-4a94-46e6-a3c7-782a6c819e89.png)

* ID Embedding을 적용하지 않은 경우    
![image](https://user-images.githubusercontent.com/74829786/177868684-1ef4fbeb-771d-4435-8844-ca24e6a7ccf8.png)

* ID Embedding을 적용한 경우가 그렇지 않은 경우보다 성능이 더 뛰어나다는 것을 확인할 수 있습니다.    
![image](https://user-images.githubusercontent.com/74829786/177868219-7c2e4a80-b301-401e-aafa-97fe1669eff7.png)

***

### 💡 실행 방법

#### 1. Data Preprocessing
```python
$ python preprocessing.py \
  --train_path =TRAIN_DATASET_PATH
  --test_size  =0.1
  --max_len    =MAX_TOKEN_LENGTH  # 256
```

#### 2. Training
```python
$ python train.py \
  --epochs =10
```

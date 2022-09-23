[Kaggle research] Personalized Medicine: Redefining Cancer Treatment
===
### *[Kaggle] Predict the effect of Genetic Variants to enable Personalized Medicine!*     
![image](https://user-images.githubusercontent.com/74829786/177875738-e780ded5-07b7-4b56-b2b5-d1d999bbd03f.png)
Kaggle research에서 제안된 문제로, 개인 맞춤형 의약품을 이용하기 위해 진행된 연구 프로젝트입니다.    
암 종양은 수천 개의 유전적 돌연변이를 가질 수 있습니다. 그 중에서 종양 성장에 기여하는 돌연변이와 중성 돌연변이를 구별하는 것이 이 리서치의 주요 목표입니다.    
이 리서치가 제안되었던 시점까지는 이 유전자 돌연변이에 대한 해석을 수동으로 수행했습니다.(임상 병리학자는 텍스트 기반 임상 문헌 근거를 기반으로 모든 단일 유전자 돌연변이를 수동으로 검토하고 분류해야 했습니다.)

**따라서 MSKCC는 유전적 변이를 자동으로 분류하는 머신러닝 알고리즘을 개발하기 위해 Kaggle과 협력하여 이 Competition을 시행하였습니다.**


                                                                                                                   
***

## 1. Dataset
* 데이터셋의 전체 크기는 총 3316으로 학습 데이터셋으로는 적은 수입니다.
* 하나의 문서당 최대 십만 글자까지 들어있기 때문에 최대 토큰수가 512인 BERT를 사용하기에는 무리가 있습니다.
* 따라서 **한 텍스트를 2000글자(대략 512토큰 이내)로 쪼개 새롭게 문서를 만들어** Data augmentation과 BERT max token length 문제를 동시에 해결했습니다. 그 결과 데이터셋 수는 3316에서 107352로 증가하였습니다.
```python
# 각 텍스트는 Gene, Variation 정보를 갖도록 한다.
# special token </s>로 구분지어준다.
dataset['texts'] = " </s> " + dataset['Gene'] + " </s> " + dataset['Variation'] + ' </s> ' + dataset['TEXT']
```

```python
# 데이터셋을 글자수 2000을 기준으로 모두 나눠 데이터프레임에 추가한다.
n = 2000
new_df = pd.DataFrame(columns={'ID', 'TEXT', 'NEW_TEXT', 'LABELS', 'CLASS'})
for i in range(len(dataset)):
    result = [dataset.iloc[i]['TEXT'][k * n:(k + 1) * n] for k in range((len(dataset.iloc[i]['TEXT']) + n - 1) // n )] 
    for j in range(len(result)):
        item = {'ID': dataset.iloc[i]['ID'], 'TEXT': result[j], 
                'NEW_TEXT': "</s> " + dataset.iloc[i]['Gene'] + " </s> " + dataset.iloc[i]['Variation'] + " </s> " + result[j],
                'LABELS': dataset.iloc[i]['labels'],
                'CLASS': dataset.iloc[i]['Class']}
        new_df = new_df.append(item, ignore_index=True)
```
* Class는 1, 2, 3...과 같은 형식으로 나타나있는데, one-hot encoding을 통해 레이블을 새로 정의해줍니다.
* 최종적인 Dataframe의 모습은 아래와 같습니다.
![image](https://user-images.githubusercontent.com/74829786/177877343-6aaaba2d-4ffc-4c88-997d-c3d02b15ca66.png)



* 실제로 실험을 진행해보니 data augmentation을 진행한 데이터셋을 학습시킨 모델이 그렇지 않은 모델보다 더 성능이 좋은 것을 확인할 수 있었습니다.    
![image](https://user-images.githubusercontent.com/74829786/177870405-2029e627-8adc-470a-bccd-7a7d8be5223b.png)


## 2. Model
* Biomedical text를 위한 사전학습 모델인 [BioBERT-Large v1.1](https://github.com/dmis-lab/biobert)를 사용하였습니다.
* ID Embedding을 적용한 모델과 적용하지 않은 모델을 설계하였습니다.


## 3. Experiments
```
epochs: 10
learning rate: 1e-4
optimizer: AdamW
scheduler: get_linear_schedule_with_warmup
loss fucntion: BCEWithLogitsLoss, log loss
```
* **ID Embedding을 적용한 경우**    
![image](https://user-images.githubusercontent.com/74829786/177868532-eb173fd2-4a94-46e6-a3c7-782a6c819e89.png)

* **ID Embedding을 적용하지 않은 경우**    
![image](https://user-images.githubusercontent.com/74829786/177868684-1ef4fbeb-771d-4435-8844-ca24e6a7ccf8.png)

* ID Embedding을 적용한 경우가 그렇지 않은 경우보다 성능이 더 뛰어나다는 것을 확인할 수 있습니다.    
![image](https://user-images.githubusercontent.com/74829786/177868219-7c2e4a80-b301-401e-aafa-97fe1669eff7.png)

***

### Usage

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

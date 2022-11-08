Personalized Medicine: Redefining Cancer Treatment
===
*Classifying genetic variations using BioBERT*
### *[Kaggle research] Predict the effect of Genetic Variants to enable Personalized Medicine! [[Link]](https://www.kaggle.com/competitions/msk-redefining-cancer-treatment)*     
![image](https://user-images.githubusercontent.com/74829786/177875738-e780ded5-07b7-4b56-b2b5-d1d999bbd03f.png)    
암 종양은 수천 개의 유전적 돌연변이를 갖고, 임상 병리학자들은 직접 수동으로 종양 성장에 기여하는 돌연변이와 중성 돌연변이를 분류하는 작업을 합니다. 이 과정은 매우 많은 시간과 비용이 소요됩니다. 따라서 Kaggle research는 MSKCC와 협력하여 문헌 정보를 근거로 자동으로 유전적 돌연변이를 분류하는 문제를 해결하고자 이 대회를 제안하였습니다.   
    
---

## 1. Preprocessing
* 학습 데이터셋의 전체 크기는 총 3316으로 적은 편에 속하지만 하나의 문서당 최대 십만 글자까지 들어가있기 때문에 최대 토큰 수가 512인 BERT를 사용하기에는 큰 무리가 있습니다.
* 따라서 **하나의 문서를 2000글자(대략 512토큰 이내)로 쪼개 새롭게 문서를 만들어 document ID를 부여하는 방식을 통해 Data augmentation과 BERT max token length 문제를 동시에 해결하였습니다.**
* 결과적으로 학습 데이터셋은 3316에서 107352로 증가되었고, 하나의 샘플당 512토큰을 갖도록 구성하였습니다.

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
* 총 9개의 유전적 변이 클래스를 갖고 있고, one-hot encoding을 통해 클래스 레이블을 새로 정의하였습니다.
* 최종적인 Dataframe의 모습은 아래와 같습니다.
![image](https://user-images.githubusercontent.com/74829786/177877343-6aaaba2d-4ffc-4c88-997d-c3d02b15ca66.png)

---

## 2. Model
* Biomedical text를 위한 사전학습 모델인 [BioBERT-Large v1.1](https://github.com/dmis-lab/biobert)를 사용하였습니다.
* ID Embedding을 적용한 모델과 적용하지 않은 모델을 설계하여 비교 실험을 진행하였습니다.

---

## 3. Experiments
```
epochs: 10
learning rate: 1e-4
optimizer: AdamW
scheduler: get_linear_schedule_with_warmup
loss fucntion: BCEWithLogitsLoss, log loss
```

* **Data Augmentation**
    * 실제로 실험을 진행해보니 data augmentation을 진행한 데이터셋을 학습시킨 모델이 그렇지 않은 모델보다 더 성능이 좋은 것을 확인할 수 있었습니다.    
![image](https://user-images.githubusercontent.com/74829786/177870405-2029e627-8adc-470a-bccd-7a7d8be5223b.png)

* **ID Embedding을 적용한 경우**    
![image](https://user-images.githubusercontent.com/74829786/177868532-eb173fd2-4a94-46e6-a3c7-782a6c819e89.png)

* **ID Embedding을 적용하지 않은 경우**    
![image](https://user-images.githubusercontent.com/74829786/177868684-1ef4fbeb-771d-4435-8844-ca24e6a7ccf8.png)

* ID Embedding을 적용한 경우가 그렇지 않은 경우보다 성능과 학습 부분에서 더 뛰어난 것을 확인할 수 있습니다.    
![image](https://user-images.githubusercontent.com/74829786/177868219-7c2e4a80-b301-401e-aafa-97fe1669eff7.png)

---

## 4. Run
### 4.1. Data Preprocessing
```python
$ python preprocessing.py \
  --train_path =TRAIN_DATASET_PATH
  --test_size  =0.1
  --max_len    =MAX_TOKEN_LENGTH  # 256
```

### 4.2. Training
```python
$ python train.py \
  --epochs =10
```

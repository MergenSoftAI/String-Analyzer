# TEKNOFEST 2023

Türkçe Doğal Dil İşleme araçlarının daha çok gelişmesi için yeni kütüphaneler tasarlandı. String Analyzer metin madenciliği için geliştirilmiş bir araçtır. 
Bu araç metinsel veriler üzerinde daha kolay ve daha doğru veri ön işleme modeli sunmaktadır. LSTM ve Lojistik Regresyon modeli kullanılmıştır.

## LSTM

### Confusion Matrix

LSTM için **confusion matrix**

```python
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
categories = data['target'].unique().tolist()
y_pred = model.predict(X_test)
y_pred = np.argmax(y_pred, axis=1)
y_true = np.argmax(Y_test, axis=1)
cm = confusion_matrix(y_true, y_pred)
sns.heatmap(cm, annot=True, cmap='Blues', fmt='g', xticklabels=categories, yticklabels=categories)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show() 

``````

<img src="https://github.com/MergenSoftAI/String-Analyzer/blob/main/Confusion%20Matrix.png" alt="Confusion Matrix" width="500"/>

### F1 Skor

```python
from sklearn.metrics import f1_score

# Eğitim verilerinin tahmini
y_pred_train = model.predict(X_train)
y_pred_train = np.argmax(y_pred_train, axis=1)

# F1 skoru hesaplama
f1_train = f1_score(np.argmax(Y_train, axis=1), y_pred_train, average='weighted')

# Test verilerinin tahmini
y_pred_test = model.predict(X_test)
y_pred_test = np.argmax(y_pred_test, axis=1)

# F1 skoru hesaplama
f1_test = f1_score(np.argmax(Y_test, axis=1), y_pred_test, average='weighted')

print("Train F1 score: ", f1_train)
print("Test F1 score: ", f1_test)

``````

<img src="https://github.com/MergenSoftAI/String-Analyzer/blob/main/F1%20Skor.png" alt="F1 Skor" width="500"/>

## Lojistik Regresyon

### F1 Skor
```python
from sklearn.metrics import f1_score
# Test verilerinin tahmini
y_pred = lr.predict(X_test_)
# F1 skoru hesaplama
f1 = f1_score(y_test_, y_pred, average='weighted')
print("F1 score:", f1)

``````
<img src="https://github.com/MergenSoftAI/String-Analyzer/blob/main/LR%20F1%20Skor.png" alt="F1 Skor" width="500"/>


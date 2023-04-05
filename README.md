# String-Analyzer

Türkçe Doğal Dil İşleme araçlarının daha çok gelişmesi için yeni kütüphaneler tasarlandı. String Analyzer metin madenciliği için geliştirilmiş bir araçtır. 
Bu araç metinsel veriler üzerinde daha kolay ve daha doğru veri ön işleme modeli sunmaktadır.

## Confusion Matrix

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

## F1 Skor

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

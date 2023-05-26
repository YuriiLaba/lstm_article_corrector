import json
from article_corrector import ArticleCorrector
from article_corrector_utils import evaluate

article_corrector_ = ArticleCorrector(
    "/content/drive/MyDrive/Colab Notebooks/language-recognition-school/language-modeling/assignment/train.json",
    "/content/drive/MyDrive/Colab Notebooks/language-recognition-school/language-modeling/assignment/train_labels.json",
    "/content/drive/MyDrive/Colab Notebooks/language-recognition-school/language-modeling/assignment/test.json",
    "/content/drive/MyDrive/Colab Notebooks/language-recognition-school/language-modeling/assignment/test_labels.json",
)

model = article_corrector_.train()
test_predictions = article_corrector_.make_predictions(model, article_corrector_.test_data)

with open('test_predictions.json', 'w', encoding='utf-8') as f:
    json.dump(test_predictions, f, ensure_ascii=False, indent=4)

evaluate("/content/drive/MyDrive/Colab Notebooks/language-recognition-school/language-modeling/assignment/test.json",
         "/content/drive/MyDrive/Colab Notebooks/language-recognition-school/language-modeling/assignment/test_labels.json",
         "test_predictions.json")

with open('/content/drive/MyDrive/Colab Notebooks/language-recognition-school/language-modeling/assignment/private.json', 'r') as f:
    private_data = json.load(f)

private_predictions = article_corrector_.make_predictions(model, private_data)
with open('private_predictions.json', 'w', encoding='utf-8') as f:
    json.dump(private_predictions, f, ensure_ascii=False, indent=4)




from transformers import BertModel, BertForPreTraining

model = BertForPreTraining.from_pretrained('bert-base-uncased')
print(model)

model = BertModel.from_pretrained('bert-base-uncased')
print(model)
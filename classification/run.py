import os
import sys
import logging
import json
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.utils.data import IterableDataset

from transformers import BertForSequenceClassification, Trainer, TrainingArguments
from transformers import AutoTokenizer


class CustomIterableDataset(IterableDataset):

    def __init__(self, filename):

        #Store the filename in object's memory
        fp = open(filename)
        self.data = json.load(fp)

        #And that's it, we no longer need to store the contents in the memory

    def __iter__(self):

        #Create an iterator
        return self.data



class CustomDataset(Dataset):
    # A pytorch dataset class for holding data for a text classification task.
    def __init__(self, filename):
        """ Takes as input the name of a file containing sentences with a classification label (comma separated) in each line.
        Stores the text data in a member variable X and labels in y
        """
        #Opening the file and storing its contents in a list
        with open(filename) as fp:
            self.data = json.load(fp)

    def preprocess(self, text):
        text = text.lower().strip()
        return text
    
    def __len__(self):
        return len(self.data)
   
    def __getitem__(self, index):
       """ Returns the text and labels present at the specified index of the lists.
       """
       x = self.preprocess(self.data[index]['text'])
       y = self.data[index]['label']
       return x, y


def check_file(filename):
    fp = open(train_data_file)
    train_data = json.load(fp)
    #for sample in train_data:
    #    print(sample)


def train_model(model, train_dataset, test_dataset):
    training_args = TrainingArguments(
        output_dir='./results',          # output directory
        num_train_epochs=3,              # total # of training epochs
        per_device_train_batch_size=16,  # batch size per device during training
        per_device_eval_batch_size=64,   # batch size for evaluation
        warmup_steps=500,                # number of warmup steps for learning rate scheduler
        weight_decay=0.01,               # strength of weight decay
        logging_dir='./logs',            # directory for storing logs
    )

    trainer = Trainer(
        model=model,                         # the instantiated Transformers model to be trained
        args=training_args,                  # training arguments, defined above
        train_dataset=train_dataset,         # training dataset
        eval_dataset=test_dataset            # evaluation dataset
    )

    trainer.train()
    trainer.evaluate()


if __name__ == "__main__":
    train_data_file = "sentence_classification_train.json"
    valid_data_file = "sentence_classification_valid.json"

    #train_dataset = CustomIterableDataset(train_data_file)
    dataset = CustomDataset(valid_data_file)

    dataloader = DataLoader(dataset, batch_size=3, num_workers=1)
    #for x, y in dataloader:
    #    print(x, y) 

    from datasets import load_dataset
    dataset = load_dataset('glue', 'mrpc', split='train')
    tokenizer = AutoTokenizer.from_pretrained('bert-base-cased')
    dataset = dataset.map(lambda e: tokenizer(e['sentence1'], truncation=True, padding='max_length'), batched=True)
    dataset.set_format(type='torch', columns=['input_ids', 'token_type_ids', 'attention_mask', 'label'])

    model = BertForSequenceClassification.from_pretrained("bert-base-uncased")
    train_model(model, train_dataset=dataset, test_dataset=dataset)
from model.model import PuncRestoreLabeler
from data_reader.reader import PuncRestoreReader
from typing import Dict, Iterable, List, Tuple
from sklearn.model_selection import train_test_split
from allennlp.data import (
    DataLoader,
    DatasetReader,
    Instance,
    Vocabulary,
    TextFieldTensors,
)
import torch
from allennlp.data.data_loaders import SimpleDataLoader
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer, PretrainedTransformerIndexer
from allennlp.data.tokenizers import Token, Tokenizer, WhitespaceTokenizer, PretrainedTransformerTokenizer
from allennlp.models import Model
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
from allennlp.modules.token_embedders import Embedding, PretrainedTransformerEmbedder
from allennlp.training.optimizers import HuggingfaceAdamWOptimizer
from allennlp.training.trainer import Trainer
from allennlp.training import GradientDescentTrainer

def build_dataset_reader() -> DatasetReader:
    tokenizer = PretrainedTransformerTokenizer(model_name)
    token_indexer = {'tokens': PretrainedTransformerIndexer(model_name)}
    return PuncRestoreReader(tokenizer, token_indexer)


def build_vocab(instances: Iterable[Instance]) -> Vocabulary:
    print("Building the vocabulary")
    return Vocabulary.from_instances(instances)


def build_model(vocab: Vocabulary) -> Model:
    print("Building the model")
    vocab_size = vocab.get_vocab_size("tokens")
    embedder = PretrainedTransformerEmbedder(model_name)
    encoder = BasicTextFieldEmbedder(token_embedders={'tokens': embedder})
    return PuncRestoreLabeler(vocab, embedder, encoder)


def read_data(reader: DatasetReader) -> Tuple[List[Instance], List[Instance]]:
    print("Reading data")
    data = list(reader.read(file_path="cleanwiki.txt"))
    training_data, validation_data = train_test_split(data, test_size=0.1)

    return training_data, validation_data


def build_data_loaders(
        train_data: List[Instance],
        dev_data: List[Instance],
) -> Tuple[DataLoader, DataLoader]:
    train_loader = SimpleDataLoader(train_data, 8, shuffle=True)
    dev_loader = SimpleDataLoader(dev_data, 8, shuffle=False)
    return train_loader, dev_loader


def build_trainer(
        model: Model,
        serialization_dir: str,
        train_loader: DataLoader,
        dev_loader: DataLoader,
) -> Trainer:
    parameters = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    optimizer = HuggingfaceAdamWOptimizer(parameters, lr=1e-5)  # type: ignore
    trainer = GradientDescentTrainer(
        model=model,
        serialization_dir=serialization_dir,
        data_loader=train_loader,
        validation_data_loader=dev_loader,
        num_epochs=100,
        optimizer=optimizer,
        cuda_device=cuda_device,
        distributed=True,
        patience=5,
        world_size=2,
    )
    return trainer

if __name__== '__main__':
    model_name = 'hfl/chinese-bert-wwm'
    punc_dic = {'，','。','；','？','！'}
    dataset_reader = build_dataset_reader()
    train_data, dev_data = read_data(dataset_reader)

    cuda_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    vocab = build_vocab(train_data + dev_data)
    model = build_model(vocab)
    # 多卡 没有用到
    # if torch.cuda.is_available():
    #     cuda_device = list(range(torch.cuda.device_count()))
    #     model = model.cuda(cuda_device[0])
    # else:
    #     cuda_device = -1
    train_loader, dev_loader = build_data_loaders(train_data, dev_data)
    train_loader.index_with(vocab)
    dev_loader.index_with(vocab)

    serialization_dir = "./save"
    # if not os.path.exists(serialization_dir):
    #     os.mkdir(serialization_dir)
    trainer = build_trainer(model, serialization_dir, train_loader, dev_loader)
    trainer.train()
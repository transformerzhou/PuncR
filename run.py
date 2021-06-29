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
from allennlp.common.util import JsonDict
from allennlp.predictors import Predictor
import argparse, re, os

parser = argparse.ArgumentParser(description='arg of model.')
parser.add_argument('--batch_size', type=int, default=8)
parser.add_argument('--threshold', type=float, default=0.9)
parser.add_argument('--bert_name', type=str, default="hfl/chinese-bert-wwm")
parser.add_argument('--save_dir', type=str, default="./save")
parser.add_argument('--mode', type=str, default="train")
parser.add_argument('--text_num', type=int, default="50000")


args = parser.parse_args()

class PuncPredictor(Predictor):
    def predict(self, sentence: str) -> JsonDict:
        return self.predict_json({"sentence": sentence})

    def _json_to_instance(self, json_dict: JsonDict) -> Instance:
        sentence = json_dict["sentence"]
        return self._dataset_reader.text_to_instance(sentence)

def build_dataset_reader() -> DatasetReader:
    tokenizer = PretrainedTransformerTokenizer(model_name)
    token_indexer = {'tokens': PretrainedTransformerIndexer(model_name)}
    return PuncRestoreReader(tokenizer, token_indexer, text_num=text_num)


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
    data = list(reader.read(file_path="./data/cleanwiki.txt"))
    training_data, validation_data = train_test_split(data, test_size=0.1)

    return training_data, validation_data


def build_data_loaders(
        train_data: List[Instance],
        dev_data: List[Instance],
) -> Tuple[DataLoader, DataLoader]:
    train_loader = SimpleDataLoader(train_data, batch_size, shuffle=True)
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
        # distributed=True,
        patience=5,
        world_size=2,
    )
    return trainer

def get_punc(query, predictor):
    """获取句子标签"""
    pred = predictor.predict(query)['probs'][1:-1]
    pun_sentence = [token + '，' if tag == 1 else token for token, tag in zip(list(query), pred)]
    #     print(pun_sentence)
    if pun_sentence[-1][-1] == '，':
        pun_sentence[-1] = pun_sentence[-1].replace('，', '。')
    else:
        pun_sentence.append('。')

    return ''.join(pun_sentence)

def test_punc(sentence):
    """对比原句和模型预测的句子"""
    r = "[，。；？！]"
    no_punc_sentence = re.sub(r, '', sentence)
    #     print(no_punc_sentence)
    punc_sentence = get_punc(no_punc_sentence, predictor)

    print(sentence, punc_sentence, sep='\n')

if __name__== '__main__':
    model_name = args.bert_name
    threshold = args.threshold
    batch_size = args.batch_size
    text_num = args.text_num
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
    trainer = build_trainer(model, serialization_dir, train_loader, dev_loader)
    if args.mode=='train':
        trainer.train()
    if args.mode=='pred':
        ckpt = torch.load(os.path.join(serialization_dir, "best.th"))
        model.load_state_dict(ckpt)
        predictor = PuncPredictor(model, dataset_reader)
        sentences = ['长龙骨黄耆学名是豆科黄芪属的植物。分布在天山、哈萨克斯坦、中亚以及中国大陆的新疆等地，生长于海拔米的地区，一般生长在草坡或荒闲地，目前尚未由人工引种栽培。',
                     '杏林觉醒建议由一些具公信力的专业机构，例如大律师公会作提名，特首选择接纳与否，担心会打破医委会的平衡。',
                     '沈家坑水库是中华人民共和国浙江省舟山市岱山县长涂镇境内的一座水库。。',
                     '在纵向不一致模型中，同卵双胞胎之间的差异能够被用于研究在时间点时性状之间的差异路径，然后检测各个明确假设，如一个个体与另一个体相比在性状上的数值盈余会导致其同一性状在未来的额外增量路径和或者另一方面也是很重要的一方面对其他性状的额外增亮路径和。在本例中关于抑郁症患者的运动量低于人群平均水平，其二者的关联是有因果关系的假设，能够被成功验证。如果运动能够缓解抑郁症，那么路径应该是显著的，即双胞胎中运动量更大的那个，抑郁程度应该更轻。',
                     '通常认为公司的财务现状被公布后，信息应当很快被投资者消化并反映在市场价格中。然而，长久以来被注意到，实际情况并非如此。对于那些获得了季度性较高利润的公司，他们的超额资产回报倾向于在公布盈利额度后向该方向再漂移至少六十天。类似地，报告较差的公司同样倾向于向不利方向漂移同样长的时间。这种现象被称为盈余惯性。',
                     '叙利亚军队多次试图进入该城，但黎巴嫩安全部队站在路上阻止叙利亚军队进城。']

        for sentence in sentences:
            test_punc(sentence)

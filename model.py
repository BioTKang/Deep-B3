from fastai.vision import *
from fastai.tabular import *
from fastai.text import *
from fastai.basics import *
from fastai.torch_core import *
from fastai.layers import *
from fastai.callbacks import *
from fastai.metrics import *
import torch
import torchvision


__all__ = ['ImageTabularTextLearner', 'collate_mixed', 'image_tabular_text_learner', 'normalize_custom_funcs']


class ImageTabularTextModel(nn.Module):
    def __init__(self, n_cont, encoder, vis_out=128, text_out=128):
        super().__init__()
        self.cnn_body = create_body(models.resnet50)
        nf = num_features_model(self.cnn_body) * 2
        self.cnn_head = create_head(nf,vis_out)
        self.nlp = SequentialRNN(encoder[0], PoolingLinearClassifier([400 * 3] + [text_out], [.5]))
        self.fc1 = nn.Sequential(*bn_drop_lin(vis_out + n_cont + text_out, 128, bn=True, p=.5, actn=nn.ReLU()))
        self.fc2 = nn.Sequential(*bn_drop_lin(128, 2, bn=False, p=0.05, actn=nn.Sigmoid()))

    def forward(self, img: Tensor, tab: Tensor, text: Tensor) -> Tensor:
        img_head = self.cnn_body(img)
        imgLatent = self.cnn_head(img_head)
        tabLatent = tab[1]
        textLatent = self.nlp(text)[0]
        cat_feature = torch.cat([F.relu(imgLatent), F.relu(tabLatent), F.relu(textLatent)], dim=1)
        pred = self.fc2(self.fc1(cat_feature))
        return pred

    def reset(self):
        for c in self.children():
            if hasattr(c, 'reset'): c.reset()

def collate_mixed(samples, pad_idx: int = 0):
    # Find max length of the text from the MixedItemList
    max_len = max([len(s[0].data[2]) for s in samples])

    for s in samples:
        res = np.zeros(max_len + pad_idx, dtype=np.int64)
        res[:len(s[0].data[2])] = s[0].data[2]
        s[0].data[2] = res

    return data_collate(samples)


def split_layers(model: nn.Module) -> List[nn.Module]:
    groups = [[model.cnn_body]]
    groups += [[model.cnn_head]]
    groups += [[model.nlp]]
    groups += [[model.fc1]]
    groups += [[model.fc2]]
    return groups

def _normalize_images_batch(b: Tuple[Tensor, Tensor], mean: FloatTensor, std: FloatTensor) -> Tuple[Tensor, Tensor]:
    x, y = b
    mean, std = mean.to(x[0].device), std.to(x[0].device)
    x[0] = normalize(x[0], mean, std)
    return x, y


def normalize_custom_funcs(mean: FloatTensor, std: FloatTensor, do_x: bool = True, do_y: bool = False) -> Tuple[
    Callable, Callable]:
    mean, std = tensor(mean), tensor(std)
    return (partial(_normalize_images_batch, mean=mean, std=std),
            partial(denormalize, mean=mean, std=std))


class RNNTrainerSimple(LearnerCallback):
    def __init__(self, learn: Learner, alpha: float = 0., beta: float = 0.):
        super().__init__(learn)
        self.not_min += ['raw_out', 'out']
        self.alpha, self.beta = alpha, beta

    def on_epoch_begin(self, **kwargs):
        self.learn.model.reset()


class ImageTabularTextLearner(Learner):
    def __init__(self, data: DataBunch, model: nn.Module, alpha: float = 2., beta: float = 1., **learn_kwargs):
        super().__init__(data, model, **learn_kwargs)
        self.callbacks.append(RNNTrainerSimple(self, alpha=alpha, beta=beta))
        self.split(split_layers)

def image_tabular_text_learner(data, len_cont_names, data_lm, loss_func, vis_out, text_out):
    l = text_classifier_learner(data_lm, AWD_LSTM, drop_mult=0.5)
    l.load_encoder('text_encoder')

    model = ImageTabularTextModel(len_cont_names, l.model, vis_out, text_out)

    learn = ImageTabularTextLearner(
        data,
        model,
        metrics=[accuracy],
        loss_func=loss_func,
    )
    return learn


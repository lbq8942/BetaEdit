import numpy
import torch
from datasets import load_dataset
from tqdm.auto import tqdm
import struct
import os
from util.nethook import Trace, set_requires_grad
from util.runningstats import CombinedStat, Mean, NormMean, SecondMoment, tally
from omegaconf import DictConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from .tok_dataset import (
    TokenizedDataset,
    dict_to_,
    flatten_masked_batch,
    length_collation,
)
null_numpy_value = numpy.array(
    struct.unpack(">d", struct.pack(">Q", 0xFFF8000000000002))[0], dtype=numpy.float64
)
global_load_cache_enabled = True
STAT_TYPES = {
    "mom2": SecondMoment,
    "mean": Mean,
    "norm_mean": NormMean,
}
def layer_stats(
        cfg,
        model,
        tokenizer,
        layer,
        ds_name,
        to_collect,
        sample_size=None,
        precision=None,
        batch_tokens=None,
        progress=tqdm,
        force_recompute=False,
):
    """
    Function to load or compute cached stats.
    """
    device = torch.device("cuda:{}".format(cfg.gpu) if torch.cuda.is_available() else "cpu")
    def get_ds(maxlen):
        raw_ds = load_dataset(
            ds_name,
            dict(wikitext="wikitext-103-raw-v1", wikipedia="20220301.en")[ds_name]
        )
        return TokenizedDataset(raw_ds["train"], tokenizer, maxlen=maxlen)
    batch_size = 100                                           
    assert hasattr(model.config, 'max_position_embeddings'),\
        ("the max sequence length can not be obtained by model.config.max_position_embeddings,"
         "Please obtain it on your own and specify it via mom2_maxseqlen in directory configs/llms")
    npos = model.config.max_position_embeddings
    if batch_tokens is not None and batch_tokens < npos:
        npos = batch_tokens
    if batch_tokens is None:
        batch_tokens = npos * 3                                                      
    if precision is None:
        precision = "float64"
    dtype = getattr(torch, precision)
    filename=cfg.cache_dir+"/stats/"+cfg.llms.name.replace("/","-") + "/layer-" + str(layer) + ".npz"
    ds = get_ds(npos) if not os.path.exists(filename) else None
    if progress is None:
        progress = lambda x: x
    stat = CombinedStat(**{k: STAT_TYPES[k]() for k in to_collect})
    loader = tally(
        stat,
        ds,
        cache=(filename if not force_recompute else None),
        sample_size=sample_size,
        batch_size=batch_size,
        collate_fn=length_collation(batch_tokens),
        pin_memory=True,
        random_sample=1,
        num_workers=2,
    )
    batch_count = -(-(sample_size or len(ds)) // batch_size)
    with torch.no_grad():
        for batch_group in progress(loader, total=batch_count):
            for batch in batch_group:
                batch = dict_to_(batch, device)
                with Trace(
                        model, cfg.llms.rewrite_module_tmp.format(layer), retain_input=True, retain_output=False, stop=True
                ) as tr:
                    model(**batch)
                feats = flatten_masked_batch(tr.input, batch["attention_mask"])
                feats = feats.to(dtype=dtype)
                stat.add(feats)
    return stat
def get_cov(
    cfg: DictConfig,
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    layer: int,
    mom2_dataset: str,
    mom2_n_samples: str,
    mom2_dtype: str,
    inv: bool = False,
    force_recompute: bool = False,
) -> torch.Tensor:
    """
    Retrieves covariance statistics, then computes the algebraic inverse.
    Caches result for future use.
    """
    device = torch.device("cuda:{}".format(cfg.gpu) if torch.cuda.is_available() else "cpu")
    model_name = cfg.llms.name.replace("/", "-")
    print(f"Retrieving covariance statistics for {model_name} @ layer {cfg.llms.rewrite_module_tmp.format(layer)}.")
    stat = layer_stats(
        cfg,
        model,
        tok,
        layer,
        mom2_dataset,
        to_collect=["mom2"],
        sample_size=mom2_n_samples,
        precision=mom2_dtype,
        batch_tokens=cfg.llms.mom2_maxseqlen,
        force_recompute=force_recompute,
    )
    cov=stat.mom2.moment()
    return (
        torch.inverse(cov) if inv else cov
    )
def is_null_numpy_value(v):
    """
    True if v is a 64-bit float numpy scalar NaN matching null_numpy_value.
    """
    return (
        isinstance(v, numpy.ndarray)
        and numpy.ndim(v) == 0
        and v.dtype == numpy.float64
        and numpy.isnan(v)
        and 0xFFF8000000000002 == struct.unpack(">Q", struct.pack(">d", v))[0]
    )
def unbox_numpy_null(d):
    """
    Reverses box_numpy_null, replacing null_numpy_value with None.
    Recursively descends into a dictionary replacing None values.
    """
    try:
        return {k: unbox_numpy_null(v) for k, v in d.items()}
    except Exception:
        return None if is_null_numpy_value(d) else d
def box_numpy_null(d):
    """
    Replaces None with null_numpy_value, leaving non-None values unchanged.
    Recursively descends into a dictionary replacing None values.
    """
    try:
        return {k: box_numpy_null(v) for k, v in d.items()}
    except Exception:
        return null_numpy_value if d is None else d
def resolve_state_dict(s):
    """
    Resolves a state, which can be a filename or a dict-like object.
    """
    if isinstance(s, str):
        return unbox_numpy_null(numpy.load(s))
    return s
def load_cached_state(cachefile, args, quiet=False, throw=False):
    """
    Resolves a state, which can be a filename or a dict-like object.
    """
    if not global_load_cache_enabled or cachefile is None:
        return None
    try:
        if isinstance(cachefile, dict):
            dat = cachefile
            cachefile = "state"                        
        else:
            dat = unbox_numpy_null(numpy.load(cachefile))
        for a, v in args.items():
            if a not in dat or dat[a] != v:
                if not quiet:
                    print("%s %s changed from %s to %s" % (cachefile, a, dat[a], v))
                return None
    except (FileNotFoundError, ValueError) as e:
        if throw:
            raise e
        return None
    else:
        if not quiet:
            print("Loading cached %s" % cachefile)
        return dat
def save_cached_state(cachefile, obj, args):
    """
    Saves the state_dict of the given object in a dict or npz file.
    """
    if cachefile is None:
        return
    dat = obj.state_dict()
    for a, v in args.items():
        if a in dat:
            assert dat[a] == v
        dat[a] = v
    if isinstance(cachefile, dict):
        cachefile.clear()
        cachefile.update(dat)
    else:
        os.makedirs(os.path.dirname(cachefile), exist_ok=True)
        numpy.savez(cachefile, **box_numpy_null(dat))
class Stat:
    """
    Abstract base class for a running pytorch statistic.
    """
    def __init__(self, state):
        """
        By convention, all Stat subclasses can be initialized by passing
        state=; and then they will initialize by calling load_state_dict.
        """
        self.load_state_dict(resolve_state_dict(state))
    def add(self, x, *args, **kwargs):
        """
        Observes a batch of samples to be incorporated into the statistic.
        Dimension 0 should be the batch dimension, and dimension 1 should
        be the feature dimension of the pytorch tensor x.
        """
        pass
    def load_state_dict(self, d):
        """
        Loads this Stat from a dictionary of numpy arrays as saved
        by state_dict.
        """
        pass
    def state_dict(self):
        """
        Saves this Stat as a dictionary of numpy arrays that can be
        stored in an npz or reloaded later using load_state_dict.
        """
        return {}
    def save(self, filename):
        """
        Saves this stat as an npz file containing the state_dict.
        """
        save_cached_state(filename, self, {})
    def load(self, filename):
        """
        Loads this stat from an npz file containing a saved state_dict.
        """
        self.load_state_dict(load_cached_state(filename, {}, quiet=True, throw=True))
    def to_(self, device):
        """
        Moves this Stat to the given device.
        """
        pass
    def cpu_(self):
        """
        Moves this Stat to the cpu device.
        """
        self.to_("cpu")
    def cuda_(self):
        """
        Moves this Stat to the default cuda device.
        """
        self.to_("cuda")
    def _normalize_add_shape(self, x, attr="data_shape"):
        """
        Flattens input data to 2d.
        """
        if not torch.is_tensor(x):
            x = torch.tensor(x)
        if len(x.shape) < 1:
            x = x.view(-1)
        data_shape = getattr(self, attr, None)
        if data_shape is None:
            data_shape = x.shape[1:]
            setattr(self, attr, data_shape)
        else:
            assert x.shape[1:] == data_shape
        return x.view(x.shape[0], int(numpy.prod(data_shape)))
    def _restore_result_shape(self, x, attr="data_shape"):
        """
        Restores output data to input data shape.
        """
        data_shape = getattr(self, attr, None)
        if data_shape is None:
            return x
        return x.view(data_shape * len(x.shape))
class SecondMoment(Stat):
    """
    Running computation. Use this when the entire non-centered 2nd-moment
    'covariance-like' matrix is needed, and when the whole matrix fits
    in the GPU.
    """
    def __init__(self, split_batch=True, state=None):
        if state is not None:
            return super().__init__(state)
        self.count = 0
        self.mom2 = None
        self.split_batch = split_batch
    def add(self, a):
        a = self._normalize_add_shape(a)
        if len(a) == 0:
            return
        if self.count == 0:
            self.mom2 = a.new(a.shape[1], a.shape[1]).zero_()
        batch_count = a.shape[0]
        self.count += batch_count
        self.mom2 += a.t().mm(a)
    def to_(self, device):
        if self.mom2 is not None:
            self.mom2 = self.mom2.to(device)
    def moment(self):
        return self.mom2 / self.count
    def state_dict(self):
        return dict(
            constructor=self.__module__ + "." + self.__class__.__name__ + "()",
            count=self.count,
            mom2=self.mom2.cpu().numpy(),
        )
    def load_state_dict(self, state):
        self.count = int(state["count"])
        self.mom2 = torch.from_numpy(state["mom2"])

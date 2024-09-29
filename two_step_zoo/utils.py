import torch
from torch.utils.data import DataLoader
import functools


def batch_or_dataloader(agg_func=torch.cat):
    def decorator(batch_fn):
        """
        Decorator for methods in which the first arg (after `self`) can either be
        a batch or a dataloader.

        The method should be coded for batch inputs. When called, the decorator will automatically
        determine whether the first input is a batch or dataloader and apply the method accordingly.
        """
        @functools.wraps(batch_fn)
        def batch_fn_wrapper(ref, batch_or_dataloader, y, **kwargs):
            if isinstance(batch_or_dataloader, DataLoader): # Input is a dataloader
                list_out = []
                for batch, y_batch, _ in batch_or_dataloader:
                    y_to_use = None if not ref.use_labels else y_batch.to(ref.device)
                    list_out.append(batch_fn(ref, batch.to(ref.device), y_to_use, **kwargs))

                if list_out and type(list_out[0]) in (list, tuple):
                    # Each member of list_out is a tuple/list; re-zip them and output a tuple
                    return tuple(agg_func(out) for out in zip(*list_out))
                else:
                    # Output is not a tuple
                    return agg_func(list_out)

            else: # Input is a batch
                return batch_fn(ref, batch_or_dataloader, y, **kwargs)

        return batch_fn_wrapper

    return decorator

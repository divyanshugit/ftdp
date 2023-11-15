from typing import Dict, List, Union,Sequence,Iterator

import torch
from transformers import PreTrainedTokenizer, DataCollatorForLanguageModeling
from torch.utils.data.sampler import Sampler, BatchSampler, RandomSampler
from torch.utils.data.distributed import DistributedSampler

class AuthorSampler(Sampler):
    def __init__(self, author_sampler: Sampler, author_mapping: Sequence[Sequence[int]]):
        self.author_mapping = list(author_mapping)
        self.author_sampler = author_sampler
        self.indices = [0 for _ in range(len(self.author_mapping))]

    def __len__(self) -> int:
        return len(self.author_sampler)

    def __iter__(self) -> Iterator[List[int]]:
        for batch_author_ids in self.author_sampler:
            sample_ids = [self.indices[author_id] for author_id in batch_author_ids]
            for author_id in batch_author_ids:
                self.indices[author_id] += 1
                self.indices[author_id] = self.indices[author_id] % len(self.author_mapping[author_id])
            yield [int(self.author_mapping[author_id][sample_id]) for author_id, sample_id in zip(batch_author_ids, sample_ids)]

class ShuffledAuthorSampler(AuthorSampler):
    def __init__(self, author_mapping: Sequence[Sequence[int]], batch_size: int, world_size: int) -> None:
        """
        Create batches by first shuffling the authors and then sampling the next element from the author

        :param author_mapping: A mapping where `dataset[author_mapping[i][j]]` produces the j-th sample of the i-th author in the dataset.
        :type author_mapping: Sequence[Sequence[int]]
        :param int batch_size: Batch size of the output
        """
        if world_size <= 1:
            author_sampler = BatchSampler(RandomSampler(author_mapping), batch_size=batch_size, drop_last=True)
        else:
            author_sampler = BatchSampler(DistributedSampler(author_mapping), batch_size=batch_size, drop_last=True)
        super().__init__(author_sampler, author_mapping)

class DataCollatorForPrivateCausalLanguageModeling(DataCollatorForLanguageModeling):
    def __init__(self, tokenizer: PreTrainedTokenizer):
        super().__init__(tokenizer=tokenizer, mlm=False)

    def __call__(self, examples: List[Union[List[int], torch.Tensor, Dict[str, torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        batch = super().__call__(examples)

        # Huggingface's default way of constructing position_ids is not compatible with Opacus
        # since Opacus is not able to deduce the batch size from the input. Here we manually
        # generate a position_ids tensor which has the same values as Huggingface's default tensor
        # but it is constructed in a way that is compatile with Opacus by using expand_as.
        if "position_ids" not in batch:
            input_ids = batch["input_ids"]
            batch["position_ids"] = torch.arange(
                input_ids.shape[1], dtype=torch.long, device=input_ids.device
            ).repeat(input_ids.shape[0], 1)
        return batch


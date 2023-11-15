from typing import Any, Dict, List, Optional, Union
from contextlib import contextmanager
import opacus
from opacus import PrivacyEngine
from transformers.trainer import Trainer
from transformers import training_args
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
# from torch.utils.data.sampler import Sampler
from ftdp.callback import DPCallback
from opacus.accountants import RDPAccountant
from prv_accountant import Accountant as PRVAccountant
from ftdp.data import ShuffledAuthorSampler as Sampler
from ftdp.utils import SFTTrainer

class GradSampleModule(opacus.GradSampleModule):
    """
    Little wrapper to provide `no_sync` context which is assumed by Huggingface trainer.
    We don't need to do anything in addition here
    """
    @contextmanager
    def no_sync(self):
        yield


class DPSFTTranier(SFTTrainer):
    def __init__(
        self,model,args,
        train_dataset,privacy_args,
        **kwargs: Dict
    ) -> None:

        self.train_args = args
        self.privacy_args = privacy_args

        # Sample-level DP is equivalent to mapping each sample to a unique author. 
        # if author_mapping is None:
        author_mapping = [[i] for i in range(len(train_dataset))]
        self.author_mapping = author_mapping

        if not self.privacy_args.is_initialized:
            self.privacy_args.initialize(
                sampling_probability=self.sampling_probability,
                num_steps=self.num_steps,
                num_samples=len(self.author_mapping),
            )

        model = GradSampleModule(model)

        self.rdp_accountant = RDPAccountant()
        self.prv_accountant = PRVAccountant(
            noise_multiplier=self.privacy_args.noise_multiplier,
            sampling_probability=self.sampling_probability,
            delta=self.privacy_args.target_delta,
            eps_error=0.1,
            max_compositions=self.num_steps,
        )
        # print(privacy_args)
        

        self.dp_callback = DPCallback(
            noise_multiplier=self.privacy_args.noise_multiplier,
            target_delta=self.privacy_args.target_delta,
            sampling_probability=self.sampling_probability,
            rdp_accountant=self.rdp_accountant,
            prv_accountant=self.prv_accountant
        )
        super().__init__(model=model, args=args, train_dataset=train_dataset, callbacks=[self.dp_callback], **kwargs)
        self.get_rdp_epsilon = lambda: self.rdp_accountant.get_epsilon(self.privacy_args.target_delta)  # RDP epsilon
        self.get_prv_epsilon = lambda: self.prv_accountant.compute_epsilon(self.state.global_step)[2]

    @property
    def sampling_probability(self) -> float:
        return self.train_args.per_device_train_batch_size * self.train_args.world_size * \
            self.train_args.gradient_accumulation_steps / len(self.author_mapping)

    @property
    def num_steps(self) -> int:
        return int(self.train_args.num_train_epochs * (1 / self.sampling_probability + 1))

    def create_optimizer(self):
        _ = super().create_optimizer()

        optimizer_generator = opacus.optimizers.DPOptimizer

        self.optimizer = optimizer_generator(
            optimizer=self.optimizer,
            noise_multiplier=self.privacy_args.noise_multiplier,
            max_grad_norm=self.privacy_args.per_sample_max_grad_norm,
            expected_batch_size=self.args.per_device_train_batch_size * self.args.gradient_accumulation_steps,
        )

        return self.optimizer

    def training_step(self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]) -> torch.Tensor:
        model.train()
        inputs = self._prepare_inputs(inputs)

        with self.compute_loss_context_manager():
                loss = self.compute_loss(model, inputs)

        loss.backward()

        return loss.detach()/self.args.gradient_accumulation_steps

    def _get_train_sampler(self):
        train_sampler = Sampler(
            author_mapping=self.author_mapping,
            batch_size=self.args.per_device_train_batch_size,
            world_size=self.args.world_size
        )
        return train_sampler

    def get_train_dataloader(self) -> DataLoader:
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        train_sampler = self._get_train_sampler()

        train_dataset = self.train_dataset
        
        return DataLoader(
            train_dataset,
            batch_sampler=train_sampler,
            collate_fn=self.data_collator,
            drop_last=self.args.dataloader_drop_last,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
        )


class DPTrainer(Trainer):
    def __init__(
        self,model,args,
        train_dataset,eval_dataset,privacy_args,
        **kwargs: Dict
    ) -> None:

        self.train_args = args
        self.privacy_args = privacy_args

        # Sample-level DP is equivalent to mapping each sample to a unique author. 
        # if author_mapping is None:
        author_mapping = [[i] for i in range(len(train_dataset))]
        self.author_mapping = author_mapping

        if not self.privacy_args.is_initialized:
            self.privacy_args.initialize(
                sampling_probability=self.sampling_probability,
                num_steps=self.num_steps,
                num_samples=len(self.author_mapping),
            )
        if isinstance(model, torch.nn.Module):
            model = GradSampleModule(model)
        else:
            model = model

        self.rdp_accountant = RDPAccountant()
        self.prv_accountant = PRVAccountant(
            noise_multiplier=self.privacy_args.noise_multiplier,
            sampling_probability=self.sampling_probability,
            delta=self.privacy_args.target_delta,
            eps_error=0.1,
            max_compositions=self.num_steps,
        )
        # print(privacy_args)
        

        self.dp_callback = DPCallback(
            noise_multiplier=self.privacy_args.noise_multiplier,
            target_delta=self.privacy_args.target_delta,
            sampling_probability=self.sampling_probability,
            rdp_accountant=self.rdp_accountant,
            prv_accountant=self.prv_accountant
        )
        super().__init__(model=model, args=args, train_dataset=train_dataset,eval_dataset=eval_dataset, callbacks=[self.dp_callback], **kwargs)
        self.get_rdp_epsilon = lambda: self.rdp_accountant.get_epsilon(self.privacy_args.target_delta)  # RDP epsilon
        self.get_prv_epsilon = lambda: self.prv_accountant.compute_epsilon(self.state.global_step)[2]

    @property
    def sampling_probability(self) -> float:
        return self.train_args.per_device_train_batch_size * self.train_args.world_size * \
            self.train_args.gradient_accumulation_steps / len(self.author_mapping)

    @property
    def num_steps(self) -> int:
        return int(self.train_args.num_train_epochs * (1 / self.sampling_probability + 1))

    def create_optimizer(self):
        _ = super().create_optimizer()

        optimizer_generator = opacus.optimizers.DPOptimizer

        self.optimizer = optimizer_generator(
            optimizer=self.optimizer,
            noise_multiplier=self.privacy_args.noise_multiplier,
            max_grad_norm=self.privacy_args.per_sample_max_grad_norm,
            expected_batch_size=self.args.per_device_train_batch_size * self.args.gradient_accumulation_steps,
        )

        return self.optimizer

    def training_step(self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]) -> torch.Tensor:
        model.train()
        inputs = self._prepare_inputs(inputs)

        with self.compute_loss_context_manager():
                loss = self.compute_loss(model, inputs)

        loss.backward()

        return loss.detach()/self.args.gradient_accumulation_steps

    def _get_train_sampler(self):
        train_sampler = Sampler(
            author_mapping=self.author_mapping,
            batch_size=self.args.per_device_train_batch_size,
            world_size=self.args.world_size
        )
        return train_sampler


    def get_train_dataloader(self) -> DataLoader:
        
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        train_sampler = self._get_train_sampler()

        train_dataset = self.train_dataset
        
        return DataLoader(
            train_dataset,
            batch_sampler=train_sampler,
            collate_fn=self.data_collator,
            drop_last=self.args.dataloader_drop_last,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
        )

    def get_eval_dataloader(self,eval) -> DataLoader:
        if self.eval_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        train_sampler = self._get_train_sampler()

        eval_dataset = self.eval_dataset
        
        return DataLoader(
            eval_dataset,
            collate_fn=self.data_collator,
            drop_last=self.args.dataloader_drop_last,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
        )
        
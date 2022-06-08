import os
import jax
import haiku as hk
import optax
from datetime import datetime
import pickle
import logging
import wandb
from omegaconf import OmegaConf

from haiku_template.modules.utils import split_treemap
# from src.dataloaders.dataloader import Dataset
from src.utils.utils import ModelClasses, forward_fn, rename_treemap_branches
from src.utils.tester import Tester
from collections import defaultdict
from functools import partial
from omegaconf import OmegaConf, listconfig

import numpy as np

# import graphviz
# dataset_enum = Dataset.SPRITEWORLD

LOG_INTERVAL = 1000

DONT_TEST = False
class Trainer:
    def __init__(self, model_name: str, cfg: OmegaConf,
                       ds_train, ds_test=None,
                       logging_path: str = None, loaded_model = None,
                       test_batch = None,
                       override_pretrain_partition_str=None,
                       override_param_matching_tuples=None) -> None:
        model_name = cfg.model.name
        cfg_model = cfg.model
        self.cfg = cfg_model.training 
        wandb.init(project='reln_slots', entity='slot-dynamics', config=OmegaConf.to_container(cfg_model),
                name=logging_path, force=True)
        model_class = ModelClasses[model_name]
        self.visualization_function = model_class.get_visualizers(cfg_model.training)

        self.rngseq = hk.PRNGSequence(self.cfg.rng_seed)
        self.ds_test = ds_test
        self.ds_train = ds_train
        self.test_sample = self._get_testing_batch(test_batch)
        self.early_stop =  False
        self.test_intervals = list(self.cfg.test_intervals) if isinstance(self.cfg.test_intervals, listconfig.ListConfig) else list(range(0, int(self.cfg.train_steps)+1, int(self.cfg.test_intervals)))

        # Initialize network and optimizer
        self.net = hk.transform_with_state(lambda x, debug: forward_fn(x, net=model_class, cfg=cfg_model.architecture, debug=debug))
        jitted_init = jax.jit(partial(self.net.init, debug=False))
        trainable_params, trainable_state = jitted_init(next(self.rngseq), self.test_sample)
        logging.info(f"Model has {hk.data_structures.tree_size(trainable_params)} parameters")
        
        # Initialize optimizer and loss function
        opt_init, self.opt_update =  model_class.get_optimizer(cfg_model.training)
        self.opt_state = opt_init(trainable_params)
        self.loss_fn = model_class.get_loss(cfg_model.training)
    
        # If a test set dataloader is provided, we'll initialize a tester
        self.tester = ds_test and Tester(cfg=(model_class, cfg), model=(self.net, None, self.net_state)) # Provide None params for now

    def _get_testing_batch(self, test_batch_in):
        # This function checks to see if a hand-selected testing batch is available, otherwise
        # just gets first batch from training set
        test_batch = next(self.ds_train) if test_batch_in is None else test_batch_in['image']
        return test_batch

    def train(self):
        # Wrapper for training so can delete if keyboard interruptes
        self._train()

    def _train(self):
        if not os.path.exists('./models'):
            os.mkdir('./models')
            os.mkdir('./outputs')

        step = 0
        start = datetime.now()
        logging.info("Started training at "+ start.strftime("%d %b %Y, %H:%M:%S"))
        loss_hist = defaultdict(list)
        while step <= self.cfg.train_steps and not self.early_stop:
            batch = next(self.ds_train)
            # NOTE no good reason thave params as a tuple - memory fuckery
            losses, other, self.params, self.net_state, self.opt_state = self.update(*self.params, *self.net_state, next(self.rngseq), self.opt_state, batch)
            # import tree_math as tm
            # [print(k, tm.Vector(v).sum()) for k,v in other[1].items()]
            [loss_hist[k].append(v.item()) for k,v in losses.items()]
            # exit()
            if step % 1000 == 0 or step == self.cfg.train_steps:
                params = self.params[1] if self.params[0] is None else hk.data_structures.merge(*self.params)
                state = self.net_state[1] if self.net_state[0] is None else hk.data_structures.merge(*self.net_state)

                if step % 10000 == 0 or step == self.cfg.train_steps:
                    # Save parameters
                    with open(f'./models/params_{step}.pkl', 'wb') as f:
                        pickle.dump((params, state), f)
                    with open(f'./models/optimizer_{step}.pkl', 'wb') as f:
                        pickle.dump(self.opt_state, f)

                # Apply model on test sequence for tensorboard
                if step % 5000 == 0:         
                    jitted_apply = jax.jit(lambda rng, inp: self.net.apply(params, state, rng, inp, True))
                    out, _ = jitted_apply(next(self.rngseq), self.test_sample) 

                    figures = self.visualization_function(self.test_sample, out) 
                    if figures:
                        for title, (f, caption) in figures.items():
                            wandb.log({title: wandb.Image(f, caption=caption)}, step=step)
                            f.savefig(f"./outputs/{title.lower().replace(' ', '_')}_{step}.pdf")
            
                # Full Validation
                if self.tester is not None and not DONT_TEST and step in self.test_intervals:
                    self.tester.update_params_and_state(params, state)
                    perf_metrics = self.tester.test(self.ds_test, train_step=step, debug=step==0)
                    np.save(f'./testing/perf_metrics_{step}.npy', perf_metrics)
                    self.early_stop = self._early_stopping(step, perf_metrics)
 
            if step % LOG_INTERVAL == 0:
                loss_summaries = {k: np.mean(v[-(min(len(v), LOG_INTERVAL)):]) for k, v in loss_hist.items()}
                wandb.log({k: float(v) for k,v in loss_summaries.items()}, step=step)
                if other is not None:
                    wandb.log({k:float(v) for k, v in other.items()}, step=step)
                
                losses_str = ', '.join([f'{k}: {v:.2e}' for k,v in loss_summaries.items()])
                logging.info(f"[[LOG_ACCURACY TRAIN]] Step: {step} ; Losses: {losses_str} ; Elapsed: " + self.strfdelta(datetime.now() - start))

            step += 1
        np.save(f'./outputs/loss_hist.npy', loss_hist)
        wandb.finish()
        logging.info("Finished training, Elapsed: " + self.strfdelta(datetime.now() - start))

    def _early_stopping(self, step, perf_metrics):
        avg_test_acc = sum(perf_metrics['direct']['accuracy'])/len(perf_metrics['direct']['accuracy'])
        logging.info(f"[[LOG_ACCURACY TEST]] Average Test Accuracy: {avg_test_acc}")
        if self.cfg.early_stop_threshold != -1:
            # return False
            return avg_test_acc > self.cfg.early_stop_threshold 
        return False

    # from https://stackoverflow.com/questions/8906926/formatting-timedelta-objects
    @staticmethod
    def strfdelta(tdelta):
        d = {"days": tdelta.days}
        d["hours"], rem = divmod(tdelta.seconds, 3600)
        d["minutes"], d["seconds"] = divmod(rem, 60)
        if d['days'] > 0:
            return "{days} days {hours:02d}:{minutes:02d}:{seconds:02d}".format(**d)
        if d["hours"] > 0:
            return "{hours:02d}:{minutes:02d}:{seconds:02d}".format(**d)
        return "{minutes:02d}:{seconds:02d}".format(**d)

    @partial(jax.jit, static_argnums=(0,)) 
    def update(self, frozen_params, trainable_params, frozen_state, trainable_state, rng_key, opt_state, batch):
        """Learning rule (stochastic gradient descent)."""
        train_grads, (losses, trainable_state, other) = jax.grad(self._loss, 1, has_aux=True)(frozen_params, trainable_params, 
                                                                                        frozen_state, trainable_state,
                                                                                        rng_key, batch)
        updates, opt_state = self.opt_update(train_grads, opt_state, trainable_params)
        trainable_params = optax.apply_updates(trainable_params, updates)
        # other = (other, train_grads)
        return losses, other, (frozen_params, trainable_params), (frozen_state, trainable_state), opt_state

    def _loss(self, frozen_params, trainable_params, frozen_state, trainable_state, rng_key, batch):
        params = trainable_params if frozen_params is None else hk.data_structures.merge(frozen_params, trainable_params)
        state = trainable_state if frozen_state is None else hk.data_structures.merge(frozen_state, trainable_state)
        x, state = self.net.apply(params, state, rng_key, batch, False)
        losses = self.loss_fn(x, batch, rng_key, params)
        #other should be a dict of k: values where values get logged to wandb
        return losses['total'], (losses, state, x['other'] if 'other' in x.keys() else None)

    # from pathlib import Path 

    # project_path = Path(__file__).parent.absolute()
    # logging_path = project_path / '/runs/slot_att/'
    # data_path = project_path.parent / "data" 
    def _clear_logs(self):
        # cwd = str(Path.cwd())
        # if str(logging_path) in cwd and len(cwd)>len(str(logging_path)):
        #     os.chdir(project_path)
        #     shutil.rmtree(cwd)
        return

    def print(self):
        print(hk.experimental.tabulate(self.net)(self.test_sample))
    
    def visualize(self):
        dot = hk.experimental.to_dot(self.net.apply)(self.params[1], next(self.rngseq), self.test_sample, True)
        src = graphviz.Source(dot)
        src.render('/media/home/alex/slot_dynamics/slot_attention.gv', view=True)  
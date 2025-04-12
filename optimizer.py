from typing import Callable, Iterable, Tuple

import torch
from torch.optim import Optimizer


class AdamW(Optimizer):
    def __init__(
            self,
            params: Iterable[torch.nn.parameter.Parameter],
            lr: float = 1e-3,
            betas: Tuple[float, float] = (0.9, 0.999),
            eps: float = 1e-6,
            weight_decay: float = 0.0,
            correct_bias: bool = True,
    ):
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {} - should be >= 0.0".format(lr))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter: {} - should be in [0.0, 1.0[".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter: {} - should be in [0.0, 1.0[".format(betas[1]))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {} - should be >= 0.0".format(eps))
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, correct_bias=correct_bias)
        super().__init__(params, defaults)

    def step(self, closure: Callable = None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError("Adam does not support sparse gradients, please consider SparseAdam instead")

                raise NotImplementedError()

                # State should be stored in this dictionary
                state = self.state[p]

                # Access hyperparameters from the `group` dictionary
                alpha = group["lr"]
                if "step" not in state:
                    state["step"] = 0
                state["step"] += 1
                t = state["step"] # our timesteps

                # gonna try and make the equation
                #mt = B1mt-1 + (1-B1)gt where gt is the gradient, mt = first moment (mean)
                # but its not really mean because it starts at 0 and gets updated in this funky way
                # the betas seem to control how much influence the gradients have in updating the new moment.
                #B1 is .9, so the new moment is only .1 influenced by the gradients. Its like a weighted average
                # probably means something in physics
                # oh wait yeah that makes sense. Its like the current is even more important than it would be in the mean cause its always 10% regardless of how many other values
                # because the derivative tells the slope of that change, if the previous ms were large it will styay large and vice versa
                # ok yeah this makes sense how its intuitively momentum
                # Hurrah I have cracked the case! Darian out! 🥚🍳😎
                if "m" not in state:
                    state["m"] = torch.zeros_like(p.data) # we initialize to 0s
                b1 = self.betas[0]

                mt_1 = state["m"] # the prev momentum
                mt = b1 * mt_1 + (1-b1) * grad




                # Bias correction
                # Please note that we are using the "efficient version" given in
                # https://arxiv.org/abs/1412.6980

                # ok so now we want
                # vt = b2 * vt-1 + (1-b2) * g^2
                # ok yeah this is quite similar just for the second moment
                if "v" not in state:
                    state["v"] = torch.zeros_like(p.data) # we initialize to 0s
                b2 = self.betas[1]

                vt_1 = state["v"] # the prev second moment
                vt = b2 * vt_1 + (1-b2) * grad**2


                # I think we save before tinkering
                state["m"] = mt
                state["v"] = vt
                # now looks like we correct our biases and become more compasionate scientists
                m_bar_t = mt / (1-b1**t) # hmm. This is actually a really big difference. Im not sure wh we do this. I will research it
                v_bar_t = vt / (1-b2**t)

                # ok, now we need
                # Ot = Ot-1 - a * m_hat_t / ( torch.sqrt(v_hat_t) + e. alpha is learning rate.
                # this is interesting because it seems like the updates are a lot more affected by past gradients than they are by current ones
                # ohhh v is probably velocity, that makes sense
                # oh wait no, silly Darian you! ok v is variance. I knew that
                # oh ok that makes sense. Its scalling it by accumlilated variance so that some parameters will update more than others
                # if the variance is less stable, the update will be smaller so that we reduce bouncing around
                # its like doing down a narrow canyon.
                # nailed it! woo! letzzz go
                update = m_bar_t / (v_bar_t.sqrt() + group["eps"])


                if group["weight_decay"] != 0.0:
                    update += weight_decay * p.data


                p.data -= lr * update



                # Update parameters

                # Add weight decay after the main gradient-based updates.
                # Please note that the learning rate should be incorporated into this update.

        return loss
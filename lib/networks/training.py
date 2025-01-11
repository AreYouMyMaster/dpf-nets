import os
from time import time
from sys import stdout
import torch.nn as nn
import numpy as np
import torch
from lib.networks.utils import AverageMeter, save_model
from lib.metrics.evaluation_metrics import jsd_between_point_cloud_sets
import numpy as np
#from remote_plot import plt
import matplotlib.pyplot as plt


def point_clouds(samples, mus, logvars):
    vars = torch.exp(logvars[0])
    return torch.pow(2.0 * np.pi * vars[0], -0.5) * torch.exp(
        -((samples[0] - mus[0]) ** 2) / 2 * vars[0]
    )
def transform_to_numpy(t):
    size_0 = t.size(dim = 0)
    size_1 = t.size(dim = 1)
    size_2 = t.size(dim = 2)
    return t.view(size_0, size_2, size_1).cpu().detach().numpy()

def train(
    iterator, model: nn.Module, loss_func, optimizer, scheduler, epoch, iter, **kwargs
):
    print(f"epoch {epoch}")
    num_workers = kwargs.get("num_workers")
    train_mode = kwargs.get("train_mode")
    model_name = os.path.join(
        kwargs["path2save"], "models", "DPFNets", kwargs.get("model_name")
    )

    batch_time = AverageMeter()
    data_time = AverageMeter()

    LB = AverageMeter()
    PNLL = AverageMeter()
    GNLL = AverageMeter()
    GENT = AverageMeter()

    model.train()
    torch.set_grad_enabled(True)

    end = time()
    loss_values = []
    for i, batch in enumerate(iterator):
        if iter + i >= len(iterator):
            break
        data_time.update(time() - end)
        scheduler(optimizer, epoch, iter + i)

        g_clouds = batch["cloud"].cuda(non_blocking=True)
        p_clouds = batch["eval_cloud"].cuda(non_blocking=True)

        model.mode = "training"
        if train_mode == "p_rnvp_mc_g_rnvp_vae":
            outputs = model(g_clouds, p_clouds)
        elif train_mode == "p_rnvp_mc_g_rnvp_vae_ic":
            images = batch["image"].cuda(non_blocking=True)
            outputs = model(g_clouds, p_clouds, images)

        samples = outputs["p_prior_samples"]

        loss, pnll, gnll, gent = loss_func(g_clouds, p_clouds, outputs)

        with torch.no_grad():
            if torch.isnan(loss):
                print("Loss is NaN! Stopping without updating the net...")
                exit()

        PNLL.update(pnll.item(), g_clouds.shape[0])
        GNLL.update(gnll.item(), g_clouds.shape[0])
        GENT.update(gent.item(), g_clouds.shape[0])
        LB.update((pnll + gnll - gent).item(), g_clouds.shape[0])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_values.append(loss.item())

        batch_time.update(time() - end)
        if (iter + i + 1) % (num_workers) == 0:
            line = "Epoch: [{0}][{1}/{2}]".format(
                epoch + 1, iter + i + 1, len(iterator)
            )
            line += "\tTime {batch_time.val:.3f} ({batch_time.avg:.3f})".format(
                batch_time=batch_time
            )
            # line += ' Data {data_time.val:.3f} ({data_time.avg:.3f})'.format(data_time=data_time)
            # line += ' LR {:.6f}'.format(optimizer.param_groups[0]['lr'])
            line += "\tLB {LB.val:.2f} ({LB.avg:.2f})".format(LB=LB)
            line += "\tPNLL {PNLL.val:.2f} ({PNLL.avg:.2f})".format(PNLL=PNLL)
            line += "\tGNLL {GNLL.val:.2f} ({GNLL.avg:.2f})".format(GNLL=GNLL)
            line += "\tGENT {GENT.val:.2f} ({GENT.avg:.2f})".format(GENT=GENT)
            line += "\n"
            stdout.write(line)
            stdout.flush()

        if i % 100 == 0:
            test_g_clouds = batch["test_points_even"].cuda(non_blocking=True)
            test_p_clouds = batch["test_points_odd"].cuda(non_blocking=True)

            model.mode = "evaluating"
            evaluate_result = model(test_g_clouds, test_p_clouds)
            samples = evaluate_result["p_prior_samples"]
            mus = evaluate_result["p_prior_mus"]
            logvars = evaluate_result["p_prior_logvars"]

            calculate_p = point_clouds(samples, mus, logvars)

            reshape_samples = transform_to_numpy(calculate_p)
            reshape_tests = transform_to_numpy(test_g_clouds)

            jsd = jsd_between_point_cloud_sets(reshape_samples, reshape_tests)
            print(f"{i}:   jsd = {jsd}")
            print(f"calculate_p size = {calculate_p.size()}")

            fig = plt.figure()
            plt.subplot(221)
            plt.plot(np.array(loss_values), 'r')

            # calculate p
            ax =fig.add_subplot(222, projection='3d')
            points = reshape_samples[0].reshape(3, -1).T
            x = points[:,0]
            y = points[:,1]
            z = points[:,2]
            ax.scatter(x,y,z, marker= '.')

            t_ax =fig.add_subplot(223, projection='3d')
            test_points = reshape_tests[0].reshape(3, -1).T
            t_x = test_points[:,0]
            t_y = test_points[:,1]
            t_z = test_points[:,2]
            t_ax.scatter(t_x,t_y,t_z, marker= '.')


            plt.show()
            plt.savefig('loss.png')

            

        end = time()

    save_model(
        {
            "epoch": epoch + 1,
            "iter": 0,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
        },
        model_name,
    )


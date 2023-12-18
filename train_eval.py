import torch.cuda
import torch.optim as optim
import time
import xlwt
from datetime import datetime
from pathlib import Path
from tensorboardX import SummaryWriter

from src.dataset.data_loader import GMDataset, get_dataloader
from src.displacement_layer import Displacement
from src.loss_func import *
from src.evaluation_metric import matching_accuracy
# from src.parallel import DataParallel
from src.utils.model_sl import load_model, save_model
from eval import eval_model
from src.lap_solvers.hungarian import hungarian
from src.utils.data_to_cuda import data_to_cuda

from src.utils.config import cfg
from pygmtools.benchmark import Benchmark
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
import numpy as np

def split_prob(prob, threshld):
    if prob.min() > threshld:
        # If prob are all larger than threshld, i.e. no noisy data, we enforce 1/100 unlabeled data
        print(
            "No estimated noisy data. Enforce the 1/100 data with small probability to be unlabeled."
        )
        threshld = np.sort(prob)[len(prob) // 100]
    pred = prob > threshld
    return pred


def split_samples(prob_A, prob_B, threshold):

    clean_mask_A = split_prob(prob_A, threshold)
    clean_mask_B = split_prob(prob_B, threshold)

    # after split clean and noisy using split_prob(), we have two lists of splits 
    clean_mask = np.zeros(prob_A.shape[0], dtype=bool) 
    clean_mask[np.logical_and(clean_mask_A == True, clean_mask_B == True)] = True

    noisy_mask = np.zeros(prob_A.shape[0], dtype=bool)
    noisy_mask[np.logical_and(clean_mask_A == False, clean_mask_B == False)] = True

    hard_mask = np.zeros(prob_A.shape[0], dtype=bool)
    hard_mask[np.logical_or(np.logical_and(clean_mask_A == True, clean_mask_B == False),
     np.logical_and(clean_mask_A == False, clean_mask_B == True))] = True
    
    return clean_mask, hard_mask, noisy_mask

def plot_gmm(epoch, gmm, X, save_path=''):
    plt.clf()
    ax = plt.gca()

    # Compute PDF of whole mixture
    x = np.linspace(0, 1, 1000)
    logprob = gmm.score_samples(x.reshape(-1, 1))
    pdf = np.exp(logprob)

    # Compute PDF for each component
    responsibilities = gmm.predict_proba(x.reshape(-1, 1))
    pdf_individual = responsibilities * pdf[:, np.newaxis]

    # Plot data histogram
    ax.hist(X, bins=100, density=True, histtype='stepfilled', color='green', alpha=0.3,
            label='Clean Samples')
    # ax.hist(X[clean_index == 0], bins=100, density=True, histtype='stepfilled', color='red', alpha=0.3, label='Noisy Samples')

    font1 = {'family': 'DejaVu Sans',
             'weight': 'normal',
             'size': 13,
             }

    if epoch == 10:
        # Plot PDF of whole model
        ax.plot(x, pdf, '-k', label='Mixture PDF')

        # Plot PDF of each component
        ax.plot(x, pdf_individual[:, 0], '--', label='Component A', color='green')
        ax.plot(x, pdf_individual[:, 1], '--', label='Component B', color='red')

    # ax.set_xlabel('Per-sample loss, epoch {}'.format(epoch), font1)
    ax.set_xlabel('Per-sample loss', font1)
    ax.set_ylabel('Density', font1)
    x_ticks = np.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    plt.xticks(x_ticks)
    plt.tick_params(labelsize=11)

    ax.legend(loc='upper right', prop=font1)

    if save_path:
        plt.savefig(save_path, dpi=300)
    else:
        plt.show()



def train_eval_model(model_A, model_B,
                     criterion,
                     optimizer_A, optimizer_B,
                     optimizer_k_A, optimizer_k_B,
                     image_dataset,
                     dataloader,
                     tfboard_writer=None,
                     benchmark=None,
                     num_epochs=25,
                     start_epoch=0,
                     xls_wb=None):
    print('Start training...')

    since = time.time()
    dataset_size = len(dataloader['train'].dataset)
    displacement = Displacement()

    device_A = next(model_A.parameters()).device
    device_B = next(model_B.parameters()).device

    print('model_A on device: {}'.format(device_A))
    print('model_B on device: {}'.format(device_B))

    checkpoint_path = Path(cfg.OUTPUT_PATH) / 'params'
    if not checkpoint_path.exists():
        checkpoint_path.mkdir(parents=True)

    model_path, optim_path, optim_k_path = '', '', ''

    
    scheduler_A = optim.lr_scheduler.MultiStepLR(optimizer_A,
                                                milestones=cfg.TRAIN.LR_STEP,
                                                gamma=cfg.TRAIN.LR_DECAY,
                                                last_epoch=-1)  # cfg.TRAIN.START_EPOCH - 1
        

    
    scheduler_B = optim.lr_scheduler.MultiStepLR(optimizer_B,
                                                milestones=cfg.TRAIN.LR_STEP,
                                                gamma=cfg.TRAIN.LR_DECAY,
                                                last_epoch=-1)  # cfg.TRAIN.START_EPOCH - 1
        

    for epoch in range(start_epoch, num_epochs):
        # Reset seed after evaluation per epoch
        if cfg.DATASET_NAME == "spair71k":
            torch.manual_seed(cfg.RANDOM_SEED + epoch + 1)
            dataloader['train'] = get_dataloader(image_dataset['train'], shuffle=True, fix_seed=False, batch_size=8) # TODO batch_size 320

        elif cfg.DATASET_NAME == "voc":
            torch.manual_seed(cfg.RANDOM_SEED + epoch + 1)
            dataloader['train'] = get_dataloader(image_dataset['train'], shuffle=True, fix_seed=False, batch_size=8) # TODO batch_size 320

        elif cfg.DATASET_NAME == "willow":
            torch.manual_seed(cfg.RANDOM_SEED + epoch + 1)
            dataloader['train'] = get_dataloader(image_dataset['train'], shuffle=True, fix_seed=False, batch_size=8) # TODO batch_size 320
        
        else:
            raise NotImplementedError
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        
        # used for divide dataset
        clean = None
        vague = None
        noisy = None
        # used for refine label
        prob_A = None
        prob_B = None
        # per epoch max iter number
        iter_num_per_epoch = len(dataloader['train'])

        print("computing losses")
        with torch.no_grad():
            model_A.eval()
            model_B.eval()
            start_time = time.time()
            losses_A = []
            losses_B = []
            count_A = 0
            max_nodes_A = 0
            count_B = 0
            max_nodes_B = 0
            iter_num = 0
            
            torch.manual_seed(cfg.RANDOM_SEED + epoch + 1) # to ensure the same training rank
            for inputs in dataloader['train']:
                if iter_num >= cfg.TRAIN.EPOCH_ITERS: # TODO del -300
                    break
                iter_num = iter_num + 1
                inputs = data_to_cuda(inputs)
                
                if 'common' in cfg.MODEL_NAME: # COMMON use the iter number to control the warmup temperature
                    outputs_A = model_A(inputs, training=True, mode="eval_loss")
                    outputs_B = model_B(inputs, training=True, mode="eval_loss")
                else:
                    raise NotImplementedError

                # compute loss
                if cfg.TRAIN.LOSS_FUNC == 'custom':
                    loss_A = outputs_A
                    loss_B = outputs_B

                    loss_A = loss_A.cpu().numpy()
                    loss_B = loss_B.cpu().numpy()

                    # print(loss_A.shape)
                    # print(loss_B.shape)
                    count_A = count_A + loss_A.shape[0]
                    max_nodes_A = max(max_nodes_A, loss_A.shape[0])
                    
                    count_B = count_B + loss_B.shape[0]
                    max_nodes_B = max(max_nodes_B, loss_B.shape[0])

                    losses_A.append(loss_A)
                    losses_B.append(loss_B)

                    # print(loss_A)
                    # print(loss_B)
                else:
                    raise NotImplementedError
                
                print("\033[A iter_num is {}/{} {} {} node samples loss computing ... {:.3f}s past".format(iter_num, iter_num_per_epoch, count_A, count_B, time.time()-start_time))

            assert count_A == count_B
            assert max_nodes_A == max_nodes_B
            print("the num of losses is {}".format(count_A))
            end_time = time.time()
            print("loss computed, using time {:.3f}s".format(end_time - start_time))
            tmp_A = np.zeros((len(image_dataset['train']), max_nodes_A))
            tmp_B = np.zeros((len(image_dataset['train']), max_nodes_B))

            mask_A = np.zeros((len(image_dataset['train']), max_nodes_A), dtype=bool)
            mask_B = np.zeros((len(image_dataset['train']), max_nodes_B), dtype=bool)


            for i in range(len(losses_A)):
                for j in range(len(losses_A[i])):
                    tmp_A[i][j] = losses_A[i][j]
                    mask_A[i][j] = True

            for i in range(len(losses_B)):
                for j in range(len(losses_B[i])):
                    tmp_B[i][j] = losses_B[i][j]
                    mask_B[i][j] = True

            print("convert loss into 2D matrix in {:.3f}s".format(time.time()-end_time))

            # print(tmp_A[0])
            # print(tmp_B[9])
            
            losses_A = tmp_A
            losses_B = tmp_B

            assert (mask_A == mask_B).all()
            # print(losses_A.shape)
            # print(losses_B.shape)
            # print(losses_A[mask_A].shape)
            # print(losses_B[mask_B].shape)

            loss_A_flaten = losses_A.reshape(-1, 1)
            loss_B_flaten = losses_B.reshape(-1, 1)

            mask_A_flaten = mask_A.reshape(-1, 1)
            mask_B_flaten = mask_B.reshape(-1, 1)

            losses_A_masked = loss_A_flaten[mask_A_flaten]
            losses_B_masked = loss_B_flaten[mask_B_flaten]

            # NaN, because of dividing by 0 (losses_A.max() - losses_A.min()) is a very small value
            input_loss_A = (losses_A_masked - np.min(losses_A_masked)) / (np.max(losses_A_masked) - np.min(losses_A_masked))
            input_loss_B = (losses_B_masked - np.min(losses_B_masked)) / (np.max(losses_B_masked) - np.min(losses_B_masked))

            print("Net A has a num of {} for GMM".format(input_loss_A.shape))
            print("Net B has a num of {} for GMM".format(input_loss_B.shape))  

            # before put into gmm must reshape (-1,1)
            input_loss_A = input_loss_A.reshape(-1, 1)
            input_loss_B = input_loss_B.reshape(-1, 1)       

            print("\nFitting GMM ...")
            # fit a two-component GMM to the loss
            gmm_A = GaussianMixture(n_components=2, max_iter=10, tol=1e-2, reg_covar=5e-4)
            gmm_A.fit(input_loss_A)
            prob_A = gmm_A.predict_proba(input_loss_A)
            prob_A = prob_A[:, gmm_A.means_.argmin()] # The probability of belonging to the first group of components (have smaller loss).

            gmm_B = GaussianMixture(n_components=2, max_iter=10, tol=1e-2, reg_covar=5e-4)
            gmm_B.fit(input_loss_B)
            prob_B = gmm_B.predict_proba(input_loss_B)
            prob_B = prob_B[:, gmm_B.means_.argmin()]

            print("The length of computed loss of network A is: {}".format(len(input_loss_A)))
            print("The length of computed loss of network B is: {}".format(len(input_loss_B)))

            plot_gmm(epoch, gmm_A, input_loss_A, cfg.OUTPUT_PATH + "/" + cfg.MODEL_NAME + '_' + cfg.DATASET_NAME + '_epoch_' + str(epoch) + '_gmm_A.png')
            plot_gmm(epoch, gmm_B, input_loss_B, cfg.OUTPUT_PATH + "/" + cfg.MODEL_NAME + '_' + cfg.DATASET_NAME + '_epoch_' + str(epoch) + '_gmm_B.png')

            clean, vague, noisy = split_samples(prob_A, prob_B, 0.5)  

            prob_A = data_to_cuda(torch.Tensor(prob_A))
            prob_B = data_to_cuda(torch.Tensor(prob_B)) 

            print("clean split has {} pairs".format(np.sum(clean)))
            print("hard split has {} pairs".format(np.sum(vague)))
            print("noisy split has {} pairs".format(np.sum(noisy)))
            
            # clean ratio 
            ratio = np.sum(clean) / (np.sum(clean)+np.sum(vague)+np.sum(noisy))

            # convert 1D mask to 2D mask for training ? or not ?

            # return

        #     print(prob_A)
        #     print(prob_B)

        # print(prob_A)
        # print(prob_B)

        model_A.train()  # Set model to training mode
        model_B.train()  # Set model to training mode

        print('lr = ' + ', '.join(['{:.2e}'.format(x['lr']) for x in optimizer_A.param_groups]))
        print('lr = ' + ', '.join(['{:.2e}'.format(x['lr']) for x in optimizer_B.param_groups]))
        

        epoch_loss = 0.0
        running_loss = 0.0
        running_ks_loss = 0.0
        running_ks_error = 0
        running_since = time.time()
        iter_num = 0

        begin_offset = 0 
        # Iterate over data.
        torch.manual_seed(cfg.RANDOM_SEED + epoch + 1) # to ensure the same training rank
        for inputs in dataloader['train']:
            if iter_num >= cfg.TRAIN.EPOCH_ITERS:
                break
            iter_num = iter_num + 1
            inputs = data_to_cuda(inputs)

            # zero the parameter gradients
            optimizer_A.zero_grad()
            optimizer_B.zero_grad()

            with torch.no_grad():
                # refine labels
                # print("refine labels...")
                # clean data
                predict_A = model_A(inputs, training=True, mode="predict")
                predict_B = model_B(inputs, training=True, mode="predict")

                # print(prob_A)
                # print(prob_B)

                # print(len_A)
                # print(len_B)
                len_A = predict_A.size(1)
                len_B = predict_B.size(1)

                assert len_A == len_B

                target_A = data_to_cuda(torch.zeros(len_A))
                target_B = data_to_cuda(torch.zeros(len_B))

                # print("prob_A", prob_A[begin_offset:begin_offset+len_A])
                # print("prob_B", prob_B[begin_offset:begin_offset+len_B])

                # print((prob_B[begin_offset:begin_offset+len_B]).size())
                # print((torch.mul(prob_B[begin_offset:begin_offset+len_B], 1)).size())
                # print(((1 - prob_B[begin_offset:begin_offset+len_B])).size())
                # print(torch.mul((1 - prob_B[begin_offset:begin_offset+len_B]), predict_A).size())

                predict_A = predict_A.squeeze(0)
                predict_B = predict_B.squeeze(0)
                
                targets_c_A = torch.mul(prob_B[begin_offset:begin_offset+len_B], 1) + \
                      torch.mul((1 - prob_B[begin_offset:begin_offset+len_B]), predict_A)
                # print(targets_c_A.size())
                # targets_c_A = targets_c_A.squeeze(0)
                # print(targets_c_A.size())
                targets_c_B = torch.mul(prob_A[begin_offset:begin_offset+len_A], 1) + \
                    torch.mul((1 - prob_A[begin_offset:begin_offset+len_A]), predict_B)
                # targets_c_B = targets_c_B.squeeze(0)

                # vague data
                targets_h_A = torch.mul((prob_A[begin_offset:begin_offset+len_A] + \
                                         prob_B[begin_offset:begin_offset+len_B]) / 2, 1) + \
                      torch.mul((1 - (prob_A[begin_offset:begin_offset+len_A] + \
                                      prob_B[begin_offset:begin_offset+len_B]) / 2), predict_A)
                # targets_h_A = targets_h_A.squeeze(0)
                
                targets_h_B = torch.mul((prob_A[begin_offset:begin_offset+len_A] + \
                                         prob_B[begin_offset:begin_offset+len_B]) / 2, 1) + \
                      torch.mul((1 - (prob_A[begin_offset:begin_offset+len_A] + \
                                      prob_B[begin_offset:begin_offset+len_B]) / 2), predict_B)
                # targets_h_B = targets_h_B.squeeze(0)

                # noisy data
                targets_u = (predict_A + predict_B) / 2

                filter_A = (np.mean(targets_c_A.cpu().numpy()) * len(clean) + np.mean(targets_h_A.cpu().numpy()) * \
                    len(vague) + np.mean(targets_u.cpu().numpy()) * len(noisy)) / (len(clean) \
                                                                                   + len(vague) + len(noisy))
                
                filter_B = (np.mean(targets_c_B.cpu().numpy()) * len(clean) + np.mean(targets_h_B.cpu().numpy()) * \
                    len(vague) + np.mean(targets_u.cpu().numpy()) * len(noisy)) / (len(clean) \
                                                                                   + len(vague) + len(noisy))

                mask_clean = clean[begin_offset:begin_offset+len_A]
                mask_vague = vague[begin_offset:begin_offset+len_A]
                mask_noisy = noisy[begin_offset:begin_offset+len_A]
                # print(mask_clean)
                # print(mask_vague)
                # print(mask_noisy)

                # print(target_A[mask_clean].size())
                # print(targets_c_A.size())

                # print(target_A[mask_clean])
                # print(targets_c_A[mask_clean])

                target_A[mask_clean] = targets_c_A[mask_clean]
                target_A[mask_vague] = targets_h_A[mask_vague]
                target_A[mask_noisy] = targets_u[mask_noisy]

                target_B[mask_clean] = targets_c_B[mask_clean]
                target_B[mask_vague] = targets_h_B[mask_vague]
                target_B[mask_noisy] = targets_u[mask_noisy]

                
                begin_offset = begin_offset+len_A

                # print(target_A)
                # print(target_B)
                # if ratio > 0.85:
                #     filter_c_A = 0
                #     filter_h_A = 0
                #     filter_c_B = 0
                #     filter_h_B = 0
                # else:
                #     filter_c_A = max(10-epoch, filter_A)
                #     filter_h_A = filter_A

                #     filter_c_B = max(10-epoch, filter_B)
                #     filter_h_B = filter_B

            with torch.set_grad_enabled(True):
                
                # forward A
                if 'common' in cfg.MODEL_NAME: # COMMON use the iter number to control the warmup temperature
                    outputs_A = model_A(inputs, training=True, iter_num=iter_num, epoch=epoch, labels=target_A, mode="train", filter=filter_A)
                    # outputs_B = model_B(inputs, training=True, iter_num=iter_num, epoch=epoch, labels=target_B, mode="train", filter=filter_B)
                else:
                    raise NotImplementedError
                if cfg.PROBLEM.TYPE == '2GM':
                    assert 'ds_mat' in outputs_A
                    assert 'perm_mat' in outputs_A
                    assert 'gt_perm_mat' in outputs_A

                    # assert 'ds_mat' in outputs_B
                    # assert 'perm_mat' in outputs_B
                    # assert 'gt_perm_mat' in outputs_B

                    # compute loss
                    if cfg.TRAIN.LOSS_FUNC == 'custom':
                        loss_A = outputs_A['loss']
                        # loss_B = outputs_B['loss']
                    else:
                        raise NotImplementedError

                    # compute accuracy
                    acc_A = matching_accuracy(outputs_A['perm_mat'], outputs_A['gt_perm_mat'], outputs_A['ns'], idx=0)
                    # acc_B = matching_accuracy(outputs_B['perm_mat'], outputs_B['gt_perm_mat'], outputs_B['ns'], idx=0)

                # backward + optimize
                if cfg.FP16 and False:
                    with amp.scale_loss(loss, optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    
                    loss = loss_A
                    loss.backward()
                    
                    if iter_num % cfg.STATISTIC_STEP == 0:
                        print("epoch {} iter {}/{} Net A has {} samples predicted. Net A loss is {}".format(epoch, iter_num, iter_num_per_epoch, len_A, loss.item()))
                    # print("epoch {} iter {}/{} Net B has {} samples predicted. Net B loss is {}".format(epoch, iter_num, iter_num_per_epoch, len_B, loss_B.item()))
                
                optimizer_A.step()
                # optimizer_B.step()

            with torch.set_grad_enabled(True):

                # forward B
                if 'common' in cfg.MODEL_NAME: # COMMON use the iter number to control the warmup temperature
                    # outputs_A = model_A(inputs, training=True, iter_num=iter_num, epoch=epoch, labels=target_A, mode="train", filter=filter_A)
                    outputs_B = model_B(inputs, training=True, iter_num=iter_num, epoch=epoch, labels=target_B, mode="train", filter=filter_B)
                else:
                    raise NotImplementedError
                if cfg.PROBLEM.TYPE == '2GM':
                    # assert 'ds_mat' in outputs_A
                    # assert 'perm_mat' in outputs_A
                    # assert 'gt_perm_mat' in outputs_A

                    assert 'ds_mat' in outputs_B
                    assert 'perm_mat' in outputs_B
                    assert 'gt_perm_mat' in outputs_B

                    # compute loss
                    if cfg.TRAIN.LOSS_FUNC == 'custom':
                        # loss_A = outputs_A['loss']
                        loss_B = outputs_B['loss']
                    else:
                        raise NotImplementedError

                    # compute accuracy
                    # acc_A = matching_accuracy(outputs_A['perm_mat'], outputs_A['gt_perm_mat'], outputs_A['ns'], idx=0)
                    acc_B = matching_accuracy(outputs_B['perm_mat'], outputs_B['gt_perm_mat'], outputs_B['ns'], idx=0)

                # backward + optimize
                if cfg.FP16 and False:
                    with amp.scale_loss(loss, optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    
                    loss = loss_B
                    loss.backward()
                    
                    if iter_num % cfg.STATISTIC_STEP == 0:
                    # print("epoch {} iter {}/{} Net A has {} samples predicted. Net A loss is {}".format(epoch, iter_num, iter_num_per_epoch, len_A, loss_A.item()))
                        print("epoch {} iter {}/{} Net B has {} samples predicted. Net B loss is {}".format(epoch, iter_num, iter_num_per_epoch, len_B, loss.item()))
                
                # optimizer_A.step()
                optimizer_B.step()

                batch_num = inputs['batch_size']

                # continue

                # tfboard writer
                loss_dict = dict()
                loss_dict['loss'] = loss_A.item() + loss_B.item()
                # tfboard_writer.add_scalars('loss', loss_dict, epoch * cfg.TRAIN.EPOCH_ITERS + iter_num)

                accdict_A = dict()
                accdict_A['matching accuracy'] = torch.mean(acc_A)

                accdict_B = dict()
                accdict_B['matching accuracy'] = torch.mean(acc_B)
                # tfboard_writer.add_scalars(
                #     'training accuracy',
                #     accdict,
                #     epoch * cfg.TRAIN.EPOCH_ITERS + iter_num
                # )

                # statistics
                running_loss += (loss_A.item() + loss_B.item()) * batch_num
                epoch_loss += (loss_A.item() + loss_B.item()) * batch_num

                if iter_num % cfg.STATISTIC_STEP == 0:
                    running_speed = cfg.STATISTIC_STEP * batch_num / (time.time() - running_since)
                    print('Epoch {:<4} Iteration {:<4} {:>4.2f}sample/s Loss={:<8.4f} Ks_Loss={:<8.4f} Ks_Error={:<8.4f}'
                          .format(epoch, iter_num, running_speed, running_loss / cfg.STATISTIC_STEP / batch_num, running_ks_loss / cfg.STATISTIC_STEP / batch_num, running_ks_error / cfg.STATISTIC_STEP / batch_num))
                    # tfboard_writer.add_scalars(
                    #     'speed',
                    #     {'speed': running_speed},
                    #     epoch * cfg.TRAIN.EPOCH_ITERS + iter_num
                    # )

                    # tfboard_writer.add_scalars(
                    #     'learning rate',
                    #     {'lr_{}'.format(i): x['lr'] for i, x in enumerate(optimizer_A.param_groups)},
                    #     epoch * cfg.TRAIN.EPOCH_ITERS + iter_num
                    # )

                    running_loss = 0.0
                    running_ks_loss = 0.0
                    running_ks_error = 0.0
                    running_since = time.time()

        # return

        epoch_loss = epoch_loss / cfg.TRAIN.EPOCH_ITERS / batch_num

        save_model(model_A, str(checkpoint_path / 'params_A_{:04}.pt'.format(epoch + 1)))
        save_model(model_B, str(checkpoint_path / 'params_B_{:04}.pt'.format(epoch + 1)))
        torch.save(optimizer_A.state_dict(), str(checkpoint_path / 'optim_A_{:04}.pt'.format(epoch + 1)))
        torch.save(optimizer_B.state_dict(), str(checkpoint_path / 'optim_B_{:04}.pt'.format(epoch + 1)))
        

        print('Epoch {:<4} Loss: {:.4f}'.format(epoch, epoch_loss))
        print()

        # Eval in each epoch
        if dataloader['test'].dataset.cls not in ['none', 'all', None]:
            clss = [dataloader['test'].dataset.cls]
        else:
            clss = dataloader['test'].dataset.bm.classes
        l_e = (epoch == (num_epochs - 1))

        accs_A = eval_model(model_A, clss, benchmark['test'], l_e)
        accs_B = eval_model(model_B, clss, benchmark['test'], l_e)

        acc_dict_A = {"{}".format(cls): single_acc for cls, single_acc in zip(dataloader['test'].dataset.classes, accs_A)}
        acc_dict_A['average'] = torch.mean(accs_A)

        acc_dict_B = {"{}".format(cls): single_acc for cls, single_acc in zip(dataloader['test'].dataset.classes, accs_B)}
        acc_dict_B['average'] = torch.mean(accs_B)
        # tfboard_writer.add_scalars(
        #     'Eval acc',
        #     acc_dict,
        #     (epoch + 1) * cfg.TRAIN.EPOCH_ITERS
        # )
        # wb.save(wb.__save_path)

        scheduler_A.step()
        scheduler_B.step()
        

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}h {:.0f}m {:.0f}s'
          .format(time_elapsed // 3600, (time_elapsed // 60) % 60, time_elapsed % 60))

    return model_A, model_B


if __name__ == '__main__':
    from src.utils.dup_stdout_manager import DupStdoutFileManager
    from src.utils.parse_args import parse_args
    from src.utils.print_easydict import print_easydict

    args = parse_args('Deep learning of graph matching training & evaluation code.')

    import importlib

    mod = importlib.import_module(cfg.MODULE)
    Net = mod.Net

    torch.manual_seed(cfg.RANDOM_SEED)

    dataset_len = {'train': cfg.TRAIN.EPOCH_ITERS * cfg.BATCH_SIZE, 'test': cfg.EVAL.SAMPLES}
    ds_dict = cfg[cfg.DATASET_FULL_NAME] if ('DATASET_FULL_NAME' in cfg) and (cfg.DATASET_FULL_NAME in cfg) else {}
    benchmark = {
        x: Benchmark(name=cfg.DATASET_FULL_NAME,
                     sets=x,
                     problem=cfg.PROBLEM.TYPE,
                     obj_resize=cfg.PROBLEM.RESCALE,
                     filter=cfg.PROBLEM.FILTER,
                     **ds_dict)
        for x in ('train', 'test')}

    image_dataset = {
        x: GMDataset(cfg.DATASET_FULL_NAME,
                     benchmark[x],
                     dataset_len[x],
                     cfg.PROBLEM.TRAIN_ALL_GRAPHS if x == 'train' else cfg.PROBLEM.TEST_ALL_GRAPHS,
                     cfg.TRAIN.CLASS if x == 'train' else cfg.EVAL.CLASS,
                     cfg.PROBLEM.TYPE)
        for x in ('train', 'test')}
    dataloader = {x: get_dataloader(image_dataset[x], shuffle=True, fix_seed=(x == 'test'))
                  for x in ('train', 'test')}

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model_A = Net()
    model_B = Net()

    model_A = model_A.to(device)
    model_B = model_B.to(device)

    if cfg.TRAIN.LOSS_FUNC.lower() == 'offset':
        criterion = OffsetLoss(norm=cfg.TRAIN.RLOSS_NORM)
    elif cfg.TRAIN.LOSS_FUNC.lower() == 'perm':
        criterion = PermutationLoss()
    elif cfg.TRAIN.LOSS_FUNC.lower() == 'ce':
        criterion = CrossEntropyLoss()
    elif cfg.TRAIN.LOSS_FUNC.lower() == 'focal':
        criterion = FocalLoss(alpha=.5, gamma=0.)
    elif cfg.TRAIN.LOSS_FUNC.lower() == 'hung':
        criterion = PermutationLossHung()
    elif cfg.TRAIN.LOSS_FUNC.lower() == 'hamming':
        criterion = HammingLoss()
    elif cfg.TRAIN.LOSS_FUNC.lower() == 'ilp':
        criterion = ILP_attention_loss()
    elif cfg.TRAIN.LOSS_FUNC.lower() == 'custom':
        criterion = None
        print('NOTE: You are setting the loss function as \'custom\', please ensure that there is a tensor with key '
              '\'loss\' in your model\'s returned dictionary.')
    else:
        raise ValueError('Unknown loss function {}'.format(cfg.TRAIN.LOSS_FUNC))

    optimizer_k_A = None
    optimizer_k_B = None

    if cfg.TRAIN.SEPARATE_BACKBONE_LR:
        if not cfg.TRAIN.SEPARATE_K_LR:

            print("not k params")
            backbone_A_ids = [id(item) for item in model_A.backbone_params]
            other_params_A = [param for param in model_A.parameters() if id(param) not in backbone_A_ids]

            model_A_params = [
                {'params': other_params_A},
                {'params': model_A.backbone_params, 'lr': cfg.TRAIN.BACKBONE_LR}
            ]

            backbone_B_ids = [id(item) for item in model_B.backbone_params]
            other_params_B = [param for param in model_B.parameters() if id(param) not in backbone_B_ids]

            model_B_params = [
                {'params': other_params_B},
                {'params': model_B.backbone_params, 'lr': cfg.TRAIN.BACKBONE_LR}
            ]

    else:
        model_A_params = model_A.parameters()
        model_B_params = model_B.parameters()

    if cfg.TRAIN.OPTIMIZER.lower() == 'sgd':
        optimizer_A = optim.SGD(model_A_params, lr=cfg.TRAIN.LR, momentum=cfg.TRAIN.MOMENTUM, nesterov=True)
        optimizer_B = optim.SGD(model_B_params, lr=cfg.TRAIN.LR, momentum=cfg.TRAIN.MOMENTUM, nesterov=True)
    elif cfg.TRAIN.OPTIMIZER.lower() == 'adam':
        optimizer_A = optim.Adam(model_A_params, lr=cfg.TRAIN.LR)
        optimizer_B = optim.Adam(model_B_params, lr=cfg.TRAIN.LR)
    else:
        raise ValueError('Unknown optimizer {}'.format(cfg.TRAIN.OPTIMIZER))

    if not Path(cfg.OUTPUT_PATH).exists():
        Path(cfg.OUTPUT_PATH).mkdir(parents=True)

    now_time = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    # tfboardwriter = SummaryWriter(logdir=str(Path(cfg.OUTPUT_PATH) / 'tensorboard' / 'training_{}'.format(now_time)))
    # wb = xlwt.Workbook()
    # wb.__save_path = str(Path(cfg.OUTPUT_PATH) / ('train_eval_result_' + now_time + '.xls'))

    with DupStdoutFileManager(str(Path(cfg.OUTPUT_PATH) / ('train_log_' + now_time + '.log'))) as _:
        print_easydict(cfg)
        model = train_eval_model(model_A, 
                                 model_B, 
                                 criterion, 
                                 optimizer_A, 
                                 optimizer_B, 
                                 optimizer_k_A, 
                                 optimizer_k_B, image_dataset, dataloader, tfboard_writer=None, benchmark=benchmark,
                                 num_epochs=cfg.TRAIN.NUM_EPOCHS,
                                 start_epoch=cfg.TRAIN.START_EPOCH,
                                 xls_wb=None)

    # wb.save(wb.__save_path)

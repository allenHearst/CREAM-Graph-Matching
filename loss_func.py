import torch
import torch.nn as nn
import torch.nn.functional as F
from src.lap_solvers.hungarian import hungarian
from src.lap_solvers.ILP import ILP_solver
from torch import Tensor


class PermutationLoss(nn.Module):
    r"""
    Binary cross entropy loss between two permutations, also known as "permutation loss".
    Proposed by `"Wang et al. Learning Combinatorial Embedding Networks for Deep Graph Matching. ICCV 2019."
    <http://openaccess.thecvf.com/content_ICCV_2019/papers/Wang_Learning_Combinatorial_Embedding_Networks_for_Deep_Graph_Matching_ICCV_2019_paper.pdf>`_

    .. math::
        L_{perm} =- \sum_{i \in \mathcal{V}_1, j \in \mathcal{V}_2}
        \left(\mathbf{X}^{gt}_{i,j} \log \mathbf{S}_{i,j} + (1-\mathbf{X}^{gt}_{i,j}) \log (1-\mathbf{S}_{i,j}) \right)

    where :math:`\mathcal{V}_1, \mathcal{V}_2` are vertex sets for two graphs.

    .. note::
        For batched input, this loss function computes the averaged loss among all instances in the batch.
    """
    def __init__(self):
        super(PermutationLoss, self).__init__()

    def forward(self, pred_dsmat: Tensor, gt_perm: Tensor, src_ns: Tensor, tgt_ns: Tensor) -> Tensor:
        r"""
        :param pred_dsmat: :math:`(b\times n_1 \times n_2)` predicted doubly-stochastic matrix :math:`(\mathbf{S})`
        :param gt_perm: :math:`(b\times n_1 \times n_2)` ground truth permutation matrix :math:`(\mathbf{X}^{gt})`
        :param src_ns: :math:`(b)` number of exact pairs in the first graph (also known as source graph).
        :param tgt_ns: :math:`(b)` number of exact pairs in the second graph (also known as target graph).
        :return: :math:`(1)` averaged permutation loss

        .. note::
            We support batched instances with different number of nodes, therefore ``src_ns`` and ``tgt_ns`` are
            required to specify the exact number of nodes of each instance in the batch.
        """
        batch_num = pred_dsmat.shape[0]

        pred_dsmat = pred_dsmat.to(dtype=torch.float32)

        try:
            assert torch.all((pred_dsmat >= 0) * (pred_dsmat <= 1))
            assert torch.all((gt_perm >= 0) * (gt_perm <= 1))
        except AssertionError as err:
            print(pred_dsmat)
            raise err

        loss = torch.tensor(0.).to(pred_dsmat.device)
        n_sum = torch.zeros_like(loss)
        for b in range(batch_num):
            batch_slice = [b, slice(src_ns[b]), slice(tgt_ns[b])]
            loss += F.binary_cross_entropy(
                pred_dsmat[batch_slice],
                gt_perm[batch_slice],
                reduction='sum')
            n_sum += src_ns[b].to(n_sum.dtype).to(pred_dsmat.device)

        return loss / n_sum


class CrossEntropyLoss(nn.Module):
    r"""
    Multi-class cross entropy loss between two permutations.

    .. math::
        L_{ce} =- \sum_{i \in \mathcal{V}_1, j \in \mathcal{V}_2} \left(\mathbf{X}^{gt}_{i,j} \log \mathbf{S}_{i,j}\right)

    where :math:`\mathcal{V}_1, \mathcal{V}_2` are vertex sets for two graphs.

    .. note::
        For batched input, this loss function computes the averaged loss among all instances in the batch.
    """
    def __init__(self):
        super(CrossEntropyLoss, self).__init__()

    def forward(self, pred_dsmat: Tensor, gt_perm: Tensor, src_ns: Tensor, tgt_ns: Tensor) -> Tensor:
        r"""
        :param pred_dsmat: :math:`(b\times n_1 \times n_2)` predicted doubly-stochastic matrix :math:`(\mathbf{S})`
        :param gt_perm: :math:`(b\times n_1 \times n_2)` ground truth permutation matrix :math:`(\mathbf{X}^{gt})`
        :param src_ns: :math:`(b)` number of exact pairs in the first graph (also known as source graph).
        :param tgt_ns: :math:`(b)` number of exact pairs in the second graph (also known as target graph).
        :return: :math:`(1)` averaged cross-entropy loss

        .. note::
            We support batched instances with different number of nodes, therefore ``src_ns`` and ``tgt_ns`` are
            required to specify the exact number of nodes of each instance in the batch.
        """
        batch_num = pred_dsmat.shape[0]

        pred_dsmat = pred_dsmat.to(dtype=torch.float32)

        try:
            assert torch.all((pred_dsmat >= 0) * (pred_dsmat <= 1))
            assert torch.all((gt_perm >= 0) * (gt_perm <= 1))
        except AssertionError as err:
            print(pred_dsmat)
            raise err

        loss = torch.tensor(0.).to(pred_dsmat.device)
        n_sum = torch.zeros_like(loss)
        for b in range(batch_num):
            batch_slice = [b, slice(src_ns[b]), slice(tgt_ns[b])]
            gt_index = torch.max(gt_perm[batch_slice], dim=-1).indices
            loss += F.nll_loss(
                torch.log(pred_dsmat[batch_slice]),
                gt_index,
                reduction='sum')
            n_sum += src_ns[b].to(n_sum.dtype).to(pred_dsmat.device)

        return loss / n_sum


class PermutationLossHung(nn.Module):
    r"""
    Binary cross entropy loss between two permutations with Hungarian attention. The vanilla version without Hungarian
    attention is :class:`~src.loss_func.PermutationLoss`.

    .. math::
        L_{hung} &=-\sum_{i\in\mathcal{V}_1,j\in\mathcal{V}_2}\mathbf{Z}_{ij}\left(\mathbf{X}^\text{gt}_{ij}\log \mathbf{S}_{ij}+\left(1-\mathbf{X}^{\text{gt}}_{ij}\right)\log\left(1-\mathbf{S}_{ij}\right)\right) \\
        \mathbf{Z}&=\mathrm{OR}\left(\mathrm{Hungarian}(\mathbf{S}),\mathbf{X}^\text{gt}\right)

    where :math:`\mathcal{V}_1, \mathcal{V}_2` are vertex sets for two graphs.

    Hungarian attention highlights the entries where the model makes wrong decisions after the Hungarian step (which is
    the default discretization step during inference).

    Proposed by `"Yu et al. Learning deep graph matching with channel-independent embedding and Hungarian attention.
    ICLR 2020." <https://openreview.net/forum?id=rJgBd2NYPH>`_

    .. note::
        For batched input, this loss function computes the averaged loss among all instances in the batch.

    A working example for Hungarian attention:

    .. image:: ../../images/hungarian_attention.png
    """
    def __init__(self):
        super(PermutationLossHung, self).__init__()

    def forward(self, pred_dsmat: Tensor, gt_perm: Tensor, src_ns: Tensor, tgt_ns: Tensor) -> Tensor:
        r"""
        :param pred_dsmat: :math:`(b\times n_1 \times n_2)` predicted doubly-stochastic matrix :math:`(\mathbf{S})`
        :param gt_perm: :math:`(b\times n_1 \times n_2)` ground truth permutation matrix :math:`(\mathbf{X}^{gt})`
        :param src_ns: :math:`(b)` number of exact pairs in the first graph (also known as source graph).
        :param tgt_ns: :math:`(b)` number of exact pairs in the second graph (also known as target graph).
        :return: :math:`(1)` averaged permutation loss

        .. note::
            We support batched instances with different number of nodes, therefore ``src_ns`` and ``tgt_ns`` are
            required to specify the exact number of nodes of each instance in the batch.
        """
        batch_num = pred_dsmat.shape[0]

        assert torch.all((pred_dsmat >= 0) * (pred_dsmat <= 1))
        assert torch.all((gt_perm >= 0) * (gt_perm <= 1))

        dis_pred = hungarian(pred_dsmat, src_ns, tgt_ns)
        ali_perm = dis_pred + gt_perm
        ali_perm[ali_perm > 1.0] = 1.0 # Hung
        pred_dsmat = torch.mul(ali_perm, pred_dsmat)
        gt_perm = torch.mul(ali_perm, gt_perm)
        loss = torch.tensor(0.).to(pred_dsmat.device)
        n_sum = torch.zeros_like(loss)
        for b in range(batch_num):
            loss += F.binary_cross_entropy(
                pred_dsmat[b, :src_ns[b], :tgt_ns[b]],
                gt_perm[b, :src_ns[b], :tgt_ns[b]],
                reduction='sum')
            n_sum += src_ns[b].to(n_sum.dtype).to(pred_dsmat.device)
        return loss / n_sum


class OffsetLoss(nn.Module):
    r"""
    OffsetLoss Criterion computes a robust loss function based on image pixel offset.
    Proposed by `"Zanfir et al. Deep Learning of Graph Matching. CVPR 2018."
    <http://openaccess.thecvf.com/content_cvpr_2018/html/Zanfir_Deep_Learning_of_CVPR_2018_paper.html>`_

    .. math::
        \mathbf{d}_i =& \sum_{j \in V_2} \left( \mathbf{S}_{i, j} P_{2j} \right)- P_{1i} \\
        L_{off} =& \sum_{i \in V_1} \sqrt{||\mathbf{d}_i - \mathbf{d}^{gt}_i||^2 + \epsilon}

    :math:`\mathbf{d}_i` is the displacement vector. See :class:`src.displacement_layer.Displacement` or more details

    :param epsilon: a small number for numerical stability
    :param norm: (optional) division taken to normalize the loss
    """
    def __init__(self, epsilon: float=1e-5, norm=None):
        super(OffsetLoss, self).__init__()
        self.epsilon = epsilon
        self.norm = norm

    def forward(self, d1: Tensor, d2: Tensor, mask: float=None) -> Tensor:
        """
        :param d1: predicted displacement matrix
        :param d2: ground truth displacement matrix
        :param mask: (optional) dummy node mask
        :return: computed offset loss
        """
        # Loss = Sum(Phi(d_i - d_i^gt))
        # Phi(x) = sqrt(x^T * x + epsilon)
        if mask is None:
            mask = torch.ones_like(mask)
        x = d1 - d2
        if self.norm is not None:
            x = x / self.norm

        xtx = torch.sum(x * x * mask, dim=-1)
        phi = torch.sqrt(xtx + self.epsilon)
        loss = torch.sum(phi) / d1.shape[0]

        return loss


class FocalLoss(nn.Module):
    r"""
    Focal loss between two permutations.

    .. math::
        L_{focal} =- \sum_{i \in \mathcal{V}_1, j \in \mathcal{V}_2}
        \left((1-\mathbf{S}_{i,j})^{\gamma} \mathbf{X}^{gt}_{i,j} \log \mathbf{S}_{i,j} +
        \mathbf{S}_{i,j}^{\gamma} (1-\mathbf{X}^{gt}_{i,j}) \log (1-\mathbf{S}_{i,j}) \right)

    where :math:`\mathcal{V}_1, \mathcal{V}_2` are vertex sets for two graphs, :math:`\gamma` is the focal loss
    hyper parameter.

    :param gamma: :math:`\gamma` parameter for focal loss
    :param eps: a small parameter for numerical stability

    .. note::
        For batched input, this loss function computes the averaged loss among all instances in the batch.
    """
    def __init__(self, gamma=0., eps=1e-15):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.eps = eps

    def forward(self, pred_dsmat: Tensor, gt_perm: Tensor, src_ns: Tensor, tgt_ns: Tensor) -> Tensor:
        r"""
        :param pred_dsmat: :math:`(b\times n_1 \times n_2)` predicted doubly-stochastic matrix :math:`(\mathbf{S})`
        :param gt_perm: :math:`(b\times n_1 \times n_2)` ground truth permutation matrix :math:`(\mathbf{X}^{gt})`
        :param src_ns: :math:`(b)` number of exact pairs in the first graph (also known as source graph).
        :param tgt_ns: :math:`(b)` number of exact pairs in the second graph (also known as target graph).
        :return: :math:`(1)` averaged focal loss

        .. note::
            We support batched instances with different number of nodes, therefore ``src_ns`` and ``tgt_ns`` are
            required to specify the exact number of nodes of each instance in the batch.
        """
        batch_num = pred_dsmat.shape[0]

        pred_dsmat = pred_dsmat.to(dtype=torch.float32)

        assert torch.all((pred_dsmat >= 0) * (pred_dsmat <= 1))
        assert torch.all((gt_perm >= 0) * (gt_perm <= 1))

        loss = torch.tensor(0.).to(pred_dsmat.device)
        n_sum = torch.zeros_like(loss)
        for b in range(batch_num):
            x = pred_dsmat[b, :src_ns[b], :tgt_ns[b]]
            y = gt_perm[b, :src_ns[b], :tgt_ns[b]]
            loss += torch.sum(
                - (1 - x) ** self.gamma * y * torch.log(x + self.eps)
                - x ** self.gamma * (1 - y) * torch.log(1 - x + self.eps)
            )
            n_sum += src_ns[b].to(n_sum.dtype).to(pred_dsmat.device)

        return loss / n_sum


class InnerProductLoss(nn.Module):
    r"""
    Inner product loss for self-supervised problems.

    .. math::
        L_{ce} =- \sum_{i \in \mathcal{V}_1, j \in \mathcal{V}_2} \left(\mathbf{X}^{gt}_{i,j} \mathbf{S}_{i,j}\right)

    where :math:`\mathcal{V}_1, \mathcal{V}_2` are vertex sets for two graphs.

    .. note::
        For batched input, this loss function computes the averaged loss among all instances in the batch.
    """
    def __init__(self):
        super(InnerProductLoss, self).__init__()

    def forward(self, pred_dsmat: Tensor, gt_perm: Tensor, src_ns: Tensor, tgt_ns: Tensor) -> Tensor:
        r"""
        :param pred_dsmat: :math:`(b\times n_1 \times n_2)` predicted doubly-stochastic matrix :math:`(\mathbf{S})`
        :param gt_perm: :math:`(b\times n_1 \times n_2)` ground truth permutation matrix :math:`(\mathbf{X}^{gt})`
        :param src_ns: :math:`(b)` number of exact pairs in the first graph (also known as source graph).
        :param tgt_ns: :math:`(b)` number of exact pairs in the second graph (also known as target graph).
        :return: :math:`(1)` averaged inner product loss

        .. note::
            We support batched instances with different number of nodes, therefore ``src_ns`` and ``tgt_ns`` are
            required to specify the exact number of nodes of each instance in the batch.
        """
        batch_num = pred_dsmat.shape[0]

        pred_dsmat = pred_dsmat.to(dtype=torch.float32)

        try:
            assert torch.all((gt_perm >= 0) * (gt_perm <= 1))
        except AssertionError as err:
            raise err

        loss = torch.tensor(0.).to(pred_dsmat.device)
        n_sum = torch.zeros_like(loss)
        for b in range(batch_num):
            batch_slice = [b, slice(src_ns[b]), slice(tgt_ns[b])]
            loss -= torch.sum(pred_dsmat[batch_slice] * gt_perm[batch_slice])
            n_sum += src_ns[b].to(n_sum.dtype).to(pred_dsmat.device)

        return loss / n_sum


class HammingLoss(torch.nn.Module):
    r"""
    Hamming loss between two permutations.

    .. math::
        L_{hamm} = \sum_{i \in \mathcal{V}_1, j \in \mathcal{V}_2}
        \left(\mathbf{X}_{i,j} (1-\mathbf{X}^{gt}_{i,j}) +  (1-\mathbf{X}_{i,j}) \mathbf{X}^{gt}_{i,j}\right)

    where :math:`\mathcal{V}_1, \mathcal{V}_2` are vertex sets for two graphs.

    Firstly adopted by `"Rolinek et al. Deep Graph Matching via Blackbox Differentiation of Combinatorial Solvers.
    ECCV 2020." <https://arxiv.org/abs/2003.11657>`_

    .. note::
        Hamming loss is defined between two discrete matrices, and discretization will in general truncate gradient. A
        workaround may be using the `blackbox differentiation technique <https://arxiv.org/abs/1912.02175>`_.
    """
    def __init__(self):
        super(HammingLoss, self).__init__()

    def forward(self, pred_perm: Tensor, gt_perm: Tensor) -> Tensor:
        r"""
        :param pred_perm: :math:`(b\times n_1 \times n_2)` predicted permutation matrix :math:`(\mathbf{X})`
        :param gt_perm: :math:`(b\times n_1 \times n_2)` ground truth permutation matrix :math:`(\mathbf{X}^{gt})`
        :return:
        """
        errors = pred_perm * (1.0 - gt_perm) + (1.0 - pred_perm) * gt_perm
        return errors.mean(dim=0).sum()


class ILP_attention_loss(nn.Module):
    r"""
    Integer Linear Programming (ILP) attention loss between two permutations.
    Proposed by `"Jiang et al. Graph-Context Attention Networks for Size-Varied Deep Graph Matching. CVPR 2022."
    <https://openaccess.thecvf.com/content/CVPR2022/html/Jiang_Graph-Context_Attention_Networks_for_Size-Varied_Deep_Graph_Matching_CVPR_2022_paper.html>`_

    .. math::
        L_{perm} =- \sum_{i \in \mathcal{V}_1, j \in \mathcal{V}_2}
        \left( \max\left(a_{i,j}, \mathbf{X}^{gt}_{i,j} \right) \log \mathbf{S}_{i,j} + (1-\mathbf{X}^{gt}_{i,j}) \log (1-\mathbf{S}_{i,j}) \right)

    where :math:`\mathcal{V}_1, \mathcal{V}_2` are vertex sets for two graphs, and a_{i,j} is the ILP assignment result.

    .. note::
        For batched input, this loss function computes the averaged loss among all instances in the batch.
    """
    def __init__(self, varied_size=True):
        super(ILP_attention_loss, self).__init__()
        self.varied_size = varied_size

    def forward(self, pred_dsmat: Tensor, gt_perm: Tensor, src_ns: Tensor, tgt_ns: Tensor) -> Tensor:
        r"""
        :param pred_dsmat: :math:`(b\times n_1 \times n_2)` predicted doubly-stochastic matrix :math:`(\mathbf{S})`
        :param gt_perm: :math:`(b\times n_1 \times n_2)` ground truth permutation matrix :math:`(\mathbf{X}^{gt})`
        :param src_ns: :math:`(b)` number of exact pairs in the first graph (also known as source graph).
        :param tgt_ns: :math:`(b)` number of exact pairs in the second graph (also known as target graph).
        :return: :math:`(1)` averaged permutation loss

        .. note::
            We support batched instances with different number of nodes, therefore ``src_ns`` and ``tgt_ns`` are
            required to specify the exact number of nodes of each instance in the batch.
        """
        batch_num = pred_dsmat.shape[0]

        pred_dsmat = pred_dsmat.to(dtype=torch.float32)

        try:
            assert torch.all((gt_perm >= 0) * (gt_perm <= 1))
        except AssertionError as err:
            print(pred_dsmat)
            raise err

        if self.varied_size:
            dis_pred = ILP_solver(pred_dsmat+1, src_ns+1, tgt_ns+1, dummy=True)
        else:
            dis_pred = ILP_solver(pred_dsmat, src_ns, tgt_ns)
        ali_perm = dis_pred + gt_perm
        ali_perm[ali_perm >= 1.0] = 1
        pred_dsmat = torch.mul(ali_perm, pred_dsmat)
        gt_perm = torch.mul(ali_perm, gt_perm)

        loss = torch.tensor(0.).to(pred_dsmat.device)
        n_sum = torch.zeros_like(loss)
        for b in range(batch_num):
            batch_slice = [b, slice(src_ns[b]), slice(tgt_ns[b])]
            loss += F.binary_cross_entropy(
                pred_dsmat[batch_slice],
                gt_perm[batch_slice],
                reduction='sum')
            n_sum += src_ns[b].to(n_sum.dtype).to(pred_dsmat.device)

        return loss / n_sum


class Distill_InfoNCE(torch.nn.Module):
    def __init__(self):
        super(Distill_InfoNCE, self).__init__()

    def forward(self, feature: Tensor, feature_m: Tensor, alpha: float, dynamic_temperature: Tensor,
                dynamic_temperature_m: Tensor) -> Tensor:
        graph1_feat = F.normalize(feature[0], dim=-1)
        graph2_feat = F.normalize(feature[1], dim=-1)

        # following the contrastive in "Learning Transferable Visual Models From Natural Language Supervision"
        sim_1to2 = dynamic_temperature.exp() * graph1_feat @ graph2_feat.T
        sim_2to1 = dynamic_temperature.exp() * graph2_feat @ graph1_feat.T

        # get momentum features
        with torch.no_grad():
            graph1_feat_m = F.normalize(feature_m[0], dim=-1)
            graph2_feat_m = F.normalize(feature_m[1], dim=-1)

            # momentum similiarity
            sim_1to2_m = dynamic_temperature_m.exp() * graph1_feat_m @ graph2_feat_m.T
            sim_2to1_m = dynamic_temperature_m.exp() * graph2_feat_m @ graph1_feat_m.T
            sim_1to2_m = F.softmax(sim_1to2_m, dim=1)
            sim_2to1_m = F.softmax(sim_2to1_m, dim=1)

            # online similiarity
            sim_targets = torch.zeros(sim_1to2_m.size()).to(graph1_feat.device)
            sim_targets.fill_diagonal_(1)

            # generate pseudo contrastive labels
            sim_1to2_targets = alpha * sim_1to2_m + (1 - alpha) * sim_targets
            sim_2to1_targets = alpha * sim_2to1_m + (1 - alpha) * sim_targets

        loss_i2t = -torch.sum(F.log_softmax(sim_1to2, dim=1) * sim_1to2_targets, dim=1).mean()
        loss_t2i = -torch.sum(F.log_softmax(sim_2to1, dim=1) * sim_2to1_targets, dim=1).mean()
        contrast_loss = (loss_i2t + loss_t2i) / 2
        return contrast_loss


class Distill_QuadraticContrast(torch.nn.Module):
    def __init__(self):
        super(Distill_QuadraticContrast, self).__init__()

    def normalize(self, x: Tensor):
        x = (x - x.min()) / (x.max() - x.min())
        return x

    def forward(self, feature: Tensor, feature_m: Tensor, dynamic_temperature: Tensor,
                dynamic_temperature_m: Tensor) -> Tensor:
        graph1_feat = F.normalize(feature[0], dim=-1)
        graph2_feat = F.normalize(feature[1], dim=-1)
        batch_size = graph1_feat.shape[0]

        with torch.no_grad():
            graph1_feat_m = F.normalize(feature_m[0], dim=-1)
            graph2_feat_m = F.normalize(feature_m[1], dim=-1)
            sim_1to2_m = graph1_feat_m @ graph2_feat_m.T
            w = ((torch.diag(sim_1to2_m) / sim_1to2_m.sum(dim=1)) + (
                        torch.diag(sim_1to2_m) / sim_1to2_m.sum(dim=0))) / 2
            # normalize w
            w = self.normalize(w)
            w = torch.mm(w.unsqueeze(1), w.unsqueeze(0))
            w = self.normalize(w)

        # cross-graph similarity
        sim_1to2 = dynamic_temperature.exp() * graph1_feat @ graph2_feat.T
        sim_2to1 = dynamic_temperature.exp() * graph2_feat @ graph1_feat.T
        # within-graph similarity
        sim_1to1 = dynamic_temperature.exp() * graph1_feat @ graph1_feat.T
        sim_2to2 = dynamic_temperature.exp() * graph2_feat @ graph2_feat.T
        # within-graph consistency
        within_graph_loss = (w * (sim_1to1 - sim_2to2).square()).mean() * batch_size / \
                            (dynamic_temperature.exp() * dynamic_temperature.exp()) # using batch_size to scale the loss
        # cross-graph consistency
        cross_graph_loss = (w * (sim_1to2 - sim_2to1).square()).mean() * batch_size / \
                           (dynamic_temperature.exp() * dynamic_temperature.exp())
        graph_loss = within_graph_loss + cross_graph_loss

        return graph_loss



class InfoNCELoss(nn.Module):
    """
    Compute infoNCE loss
    """
    def __init__(self, temperature=0.07):
        super(InfoNCELoss, self).__init__()

        self.temperature = temperature


    def forward(
        self,
        scores,# similarity matrix.
        labels=None, # refined labels. when mode="predict", labels must be None.
        mode="train",
        soft_label=True,   # when true, one hot is changed to soft during calculating infoNCE loss 
        smooth_label=False, # this argument is useless when soft_label is set to False
        filter=0, # if weight < filter, then weight = 0, only for soft_label=True, smooth_label=False
    ):  
        if labels is not None:
            labels = labels.t()[0]
        similarity = scores.clone() # copy a similarity as backup.
        scores = torch.exp(torch.div(scores, self.temperature))
        diagonal = scores.diag().view(scores.size(0), 1).t().to(scores.device)
        
        sum_row = scores.sum(1)
        sum_col = scores.sum(0)
        
        loss_text_retrieval = -torch.log(torch.div(diagonal, sum_row))
        loss_image_retrieval = -torch.log(torch.div(diagonal, sum_col))
        
        if mode == "predict":

            predict_text_retrieval = torch.div(scores.t(), sum_row).t()
            predict_image_retrieval = torch.div(scores, sum_col)

            p = (predict_text_retrieval + predict_image_retrieval) / 2
            p = p.diag().view(scores.size(0), 1).t().to(scores.device)
            p = p.clamp(min=0, max=1)
            
            return p
            
        elif mode == "warmup":
            # Warm up with the original infoNCE.
            return (loss_text_retrieval + loss_image_retrieval)[0].mean(), torch.Tensor([0]), torch.Tensor([0])
        elif mode == "train":
            if soft_label:
                # soft_label, two versions.
                loss_text_retrieval = -torch.log(torch.div(scores.t(), sum_row).t())
                loss_image_retrieval = -torch.log(torch.div(scores, sum_col))
                
                if smooth_label:
                    # positive pairs' weights are labels, and the remaining 
                    # sample pairs have mean scores: (1-labels) / batch_size - 1.  
                    label_smooth = ((1 - labels)/(scores.size(0)-1)) * torch.ones(scores.size()).to(scores.device)
                    labels = torch.eye(scores.size(0)).to(labels.device) * labels
                    # if labels = torch.tensor([0.7,0.8,0.9,0.4]), size is (1,4)
                    # then the sum of label_smooth‘s each column is one

                    mask = torch.eye(scores.size(0)) > 0.5
                    mask = mask.to(label_smooth.device)
                    label_smooth = label_smooth.masked_fill_(mask, 0)
                    label_smooth = labels + label_smooth


                    loss_text_retrieval = torch.mul(label_smooth.t(), loss_text_retrieval)
                    loss_image_retrieval = torch.mul(label_smooth, loss_image_retrieval)

                    return (loss_text_retrieval + loss_image_retrieval).sum() / scores.size(0), (torch.max(label_smooth + label_smooth.t() - 2 * labels, 1)[0].mean() / 2), (torch.max(label_smooth + label_smooth.t() - 2 * labels, 0)[0].mean() / 2)
                
                else:
                    # the remaining sample pairs share (1-labels) according to similarity (scores).
                    positive_weight = torch.eye(scores.size(0)).to(labels.device) * labels
                    mask = torch.eye(scores.size(0)) > 0.5
                    mask = mask.to(similarity.device)
                    similarity = similarity.masked_fill_(mask, 0)

                    sum_row = similarity.sum(1)
                    sum_col = similarity.sum(0)

                    # add a very small value (1e-6) to prevent division by 0.
                    weight_row = torch.div(similarity.t(), sum_row + 1e-6).t()
                    weight_col = torch.div(similarity, sum_col + 1e-6)

                    d1 = (1-labels).expand_as(weight_row).t()
                    d2 = (1-labels).expand_as(weight_col)
                    weight_row = torch.mul(d1, weight_row)
                    weight_col = torch.mul(d2, weight_col)

                    weight_col[weight_col < filter] = 0
                    weight_row[weight_row < filter] = 0

                    weight_row = weight_row + positive_weight
                    weight_col = weight_col + positive_weight
                    
                    loss_text_retrieval = torch.mul(loss_text_retrieval, weight_row)
                    loss_image_retrieval = torch.mul(loss_image_retrieval, weight_col)

                    return ((loss_text_retrieval + loss_image_retrieval).sum() / scores.size(0)), (torch.max(weight_row + weight_col - 2 * positive_weight, 1)[0].mean() / 2), (torch.max(weight_row + weight_col - 2 * positive_weight, 0)[0].mean() / 2)

            else:
                return (loss_text_retrieval + loss_image_retrieval)[0].mean(), torch.Tensor([0]), torch.Tensor([0])

        # two versions for eval loss. one is using the positive pair for loss computation, the other is using infoNCE loss for per sample loss computation.
        elif mode == "eval_loss":
            # using infoNCE loss for per sample loss computation.
            return (loss_text_retrieval + loss_image_retrieval)[0], torch.Tensor([0]), torch.Tensor([0])